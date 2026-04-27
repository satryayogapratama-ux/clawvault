"""
ClawVault: Local RAG Engine for OpenClaw

A self-contained knowledge base system using sentence-transformers for embeddings
and SQLite for vector storage. No external APIs required.
"""

import ipaddress
import json
import socket
import sqlite3
import time
from datetime import datetime
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import re

# ─── SSRF Protection ──────────────────────────────────────────────────────────
ALLOWED_SCHEMES = {'http', 'https'}
MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB cap

def _is_safe_url(url: str) -> tuple[bool, str]:
    """
    Validate URL to prevent SSRF attacks.
    Blocks: private IPs, link-local, loopback, metadata endpoints, non-http(s).
    Returns (is_safe, reason).
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL"

    if parsed.scheme not in ALLOWED_SCHEMES:
        return False, f"Scheme '{parsed.scheme}' not allowed (only http/https)"

    hostname = parsed.hostname
    if not hostname:
        return False, "No hostname"

    # Block known cloud metadata endpoints
    blocked_hosts = {'169.254.169.254', 'metadata.google.internal', 'metadata.aws.internal'}
    if hostname.lower() in blocked_hosts:
        return False, "Metadata endpoint blocked"

    # Resolve hostname and check IP range
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in addr_info:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False, f"Private/internal IP blocked: {ip}"
    except socket.gaierror:
        return False, f"Cannot resolve hostname: {hostname}"
    except ValueError:
        pass  # Not a valid IP string — already resolved above

    return True, "ok"

# Try importing sentence-transformers, with fallback
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("WARNING: sentence-transformers not installed. Using dummy embeddings.")

# Try importing PyPDF2, with fallback
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    print("WARNING: PyPDF2 not installed. PDF support disabled.")


@dataclass
class Document:
    """Represents an ingested document."""
    doc_id: str
    title: str
    source: str
    source_type: str  # 'pdf', 'markdown', 'text', 'url'
    content: str


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    chunk_id: str
    doc_id: str
    text: str
    start_pos: int
    end_pos: int
    embedding: Optional[np.ndarray] = None


class TextChunker:
    """Smart text chunking with respect for paragraph and code boundaries."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks, respecting paragraph and code block boundaries.
        Size is approximate token count (rough: 1 word ≈ 1.3 tokens).
        """
        chunks = []
        
        # Split by double newlines (paragraphs) first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = len(para.split())
            
            # If single paragraph is huge, split it
            if para_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = ""
                temp_tokens = 0
                
                for sent in sentences:
                    sent_tokens = len(sent.split())
                    if temp_tokens + sent_tokens > self.chunk_size:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sent
                        temp_tokens = sent_tokens
                    else:
                        temp_chunk += (" " if temp_chunk else "") + sent
                        temp_tokens += sent_tokens
                
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
                current_chunk = ""
                current_tokens = 0
            else:
                # Add paragraph to current chunk
                if current_tokens + para_tokens > self.chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = para
                    current_tokens = para_tokens
                else:
                    current_chunk += ("\n\n" if current_chunk else "") + para
                    current_tokens += para_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Add overlap
        if len(chunks) > 1:
            final_chunks = []
            for i, chunk in enumerate(chunks):
                if i == 0:
                    final_chunks.append(chunk)
                else:
                    # Add overlap from previous chunk
                    words = chunks[i-1].split()
                    overlap_text = " ".join(words[-self.overlap:]) if len(words) > self.overlap else chunks[i-1]
                    final_chunks.append(overlap_text + "\n\n" + chunk)
            return final_chunks
        
        return chunks


class EmbeddingEngine:
    """Handles text embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            except Exception as e:
                print(f"Error loading model {model_name}: {e}. Using dummy embeddings.")
                self.model = None
                self.embedding_dim = 384  # Default dimension
        else:
            self.model = None
            self.embedding_dim = 384
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts. Returns shape (len(texts), embedding_dim)."""
        if self.model:
            try:
                return self.model.encode(texts, convert_to_numpy=True)
            except Exception as e:
                print(f"Error embedding texts: {e}")
                return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
        else:
            # Dummy embeddings (deterministic hash-based)
            embeddings = []
            for text in texts:
                seed = hash(text) % (2**31)
                np.random.seed(seed)
                embeddings.append(np.random.randn(self.embedding_dim).astype(np.float32))
            return np.array(embeddings)


class VectorStore:
    """SQLite + numpy vector storage with cosine similarity search."""
    
    def __init__(self, db_path: str = "clawvault.db"):
        self.db_path = db_path
        self.embeddings = {}  # chunk_id -> embedding array
        self.chunks_by_doc = {}  # doc_id -> [chunk_ids]
        self._init_db()
        self._load_embeddings_from_db()  # load persisted embeddings into memory

    def _load_embeddings_from_db(self):
        """Load all stored embeddings from DB into memory on startup."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_id, embedding FROM embeddings")
        rows = cursor.fetchall()
        conn.close()
        for chunk_id, emb_blob in rows:
            if emb_blob:
                arr = np.frombuffer(emb_blob, dtype=np.float32).copy()
                self.embeddings[chunk_id] = arr
    
    def _init_db(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            source TEXT,
            source_type TEXT,
            content TEXT,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            text TEXT,
            start_pos INTEGER,
            end_pos INTEGER,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id TEXT PRIMARY KEY,
            embedding BLOB,
            FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
        )
        """)
        
        conn.commit()
        conn.close()
    
    def add_document(self, doc: Document, chunks: List[Chunk]):
        """Store document and its chunks with embeddings."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert document
        cursor.execute("""
        INSERT OR REPLACE INTO documents (doc_id, title, source, source_type, content)
        VALUES (?, ?, ?, ?, ?)
        """, (doc.doc_id, doc.title, doc.source, doc.source_type, doc.content))
        
        # Insert chunks
        chunk_ids = []
        for chunk in chunks:
            cursor.execute("""
            INSERT OR REPLACE INTO chunks (chunk_id, doc_id, text, start_pos, end_pos)
            VALUES (?, ?, ?, ?, ?)
            """, (chunk.chunk_id, chunk.doc_id, chunk.text, chunk.start_pos, chunk.end_pos))
            
            # Store embedding in memory
            if chunk.embedding is not None:
                self.embeddings[chunk.chunk_id] = chunk.embedding
                
                # Also store in DB as blob
                embedding_blob = chunk.embedding.tobytes()
                cursor.execute("""
                INSERT OR REPLACE INTO embeddings (chunk_id, embedding)
                VALUES (?, ?)
                """, (chunk.chunk_id, embedding_blob))
            
            chunk_ids.append(chunk.chunk_id)
        
        self.chunks_by_doc[doc.doc_id] = chunk_ids
        conn.commit()
        conn.close()
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find top-K most similar chunks using cosine similarity."""
        if not self.embeddings:
            return []
        
        results = []
        for chunk_id, embedding in self.embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-8
            )
            results.append((chunk_id, float(similarity)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_chunk_text(self, chunk_id: str) -> Optional[str]:
        """Retrieve chunk text by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def get_documents(self) -> List[Dict]:
        """List all documents."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id, title, source, source_type FROM documents ORDER BY ingested_at DESC")
        docs = [
            {"doc_id": row[0], "title": row[1], "source": row[2], "source_type": row[3]}
            for row in cursor.fetchall()
        ]
        conn.close()
        return docs
    
    def delete_document(self, doc_id: str):
        """Delete document and its chunks."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        
        # Clean up embeddings cache
        if doc_id in self.chunks_by_doc:
            for chunk_id in self.chunks_by_doc[doc_id]:
                if chunk_id in self.embeddings:
                    del self.embeddings[chunk_id]
            del self.chunks_by_doc[doc_id]
        
        conn.commit()
        conn.close()


class DocumentIngester:
    """Load documents from various sources."""
    
    @staticmethod
    def from_pdf(file_path: str) -> Optional[str]:
        """Extract text from PDF."""
        if not HAS_PYPDF2:
            return None
        
        try:
            text = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text())
            return "\n\n".join(text)
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return None
    
    @staticmethod
    def from_markdown(file_path: str) -> str:
        """Load markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def from_text(file_path: str) -> str:
        """Load plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def from_url(url: str) -> Optional[str]:
        """Fetch and extract text from URL."""
        safe, reason = _is_safe_url(url)
        if not safe:
            print(f"URL blocked ({reason}): {url}")
            return None

        try:
            response = requests.get(url, timeout=10, stream=True)
            response.raise_for_status()

            # Enforce size limit
            content_bytes = b''
            for chunk in response.iter_content(chunk_size=65536):
                content_bytes += chunk
                if len(content_bytes) > MAX_RESPONSE_BYTES:
                    print(f"URL response too large (>{MAX_RESPONSE_BYTES // 1024}KB): {url}")
                    return None

            soup = BeautifulSoup(content_bytes, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    @staticmethod
    def ingest(source: str, source_type: Optional[str] = None) -> Optional[Document]:
        """
        Load document from file or URL.
        Auto-detects type if not specified.
        """
        source_path = Path(source)
        
        # Auto-detect type
        if source_type is None:
            if source.startswith(('http://', 'https://')):
                source_type = 'url'
            elif source_path.suffix.lower() == '.pdf':
                source_type = 'pdf'
            elif source_path.suffix.lower() in ['.md', '.markdown']:
                source_type = 'markdown'
            else:
                source_type = 'text'
        
        # Load content
        content = None
        title = source if source_type == 'url' else source_path.stem
        
        if source_type == 'url':
            content = DocumentIngester.from_url(source)
        elif source_type == 'pdf':
            content = DocumentIngester.from_pdf(source)
        elif source_type == 'markdown':
            content = DocumentIngester.from_markdown(source)
        else:
            content = DocumentIngester.from_text(source)
        
        if content is None:
            return None
        
        # Generate doc_id
        doc_id = f"{source_type}_{hash(source) % (10**9)}"
        
        return Document(
            doc_id=doc_id,
            title=title,
            source=source,
            source_type=source_type,
            content=content
        )


class QueryEngine:
    """Semantic search and answer generation."""
    
    def __init__(self, embedding_engine: EmbeddingEngine, vector_store: VectorStore):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Search for relevant chunks and format as context.
        Returns: {
            'question': str,
            'chunks': [{'text': str, 'similarity': float, 'source': str}],
            'context': str
        }
        """
        # Embed query
        query_embedding = self.embedding_engine.embed([question])[0]
        
        # Search
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        chunks = []
        context_parts = []
        
        for chunk_id, similarity in results:
            text = self.vector_store.get_chunk_text(chunk_id)
            if text:
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': text,
                    'similarity': similarity
                })
                context_parts.append(f"[Chunk {chunk_id}]\n{text}")
        
        return {
            'question': question,
            'chunks': chunks,
            'context': "\n\n---\n\n".join(context_parts)
        }


class ClawVault:
    """Main RAG interface."""
    
    def __init__(self, db_path: str = "clawvault.db", model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_engine = EmbeddingEngine(model_name)
        self.vector_store = VectorStore(db_path)
        self.query_engine = QueryEngine(self.embedding_engine, self.vector_store)
        self.chunker = TextChunker(chunk_size=500, overlap=50)
    
    def add(self, source: str, source_type: Optional[str] = None) -> bool:
        """Ingest a document from file, URL, or text."""
        print(f"Ingesting {source}...")
        
        doc = DocumentIngester.ingest(source, source_type)
        if not doc:
            print(f"Failed to ingest {source}")
            return False
        
        # Chunk
        chunks_text = self.chunker.chunk(doc.content)
        
        # Embed and create chunk objects
        embeddings = self.embedding_engine.embed(chunks_text)
        chunks = []
        
        for i, (text, embedding) in enumerate(zip(chunks_text, embeddings)):
            chunk = Chunk(
                chunk_id=f"{doc.doc_id}_chunk_{i}",
                doc_id=doc.doc_id,
                text=text,
                start_pos=0,
                end_pos=len(text),
                embedding=embedding
            )
            chunks.append(chunk)
        
        # Store
        self.vector_store.add_document(doc, chunks)
        print(f"✓ Ingested: {doc.title} ({len(chunks)} chunks)")
        return True
    
    def ask(self, question: str, top_k: int = 5) -> Dict:
        """Query the knowledge base."""
        return self.query_engine.query(question, top_k=top_k)
    
    def list_docs(self) -> List[Dict]:
        """List all documents."""
        return self.vector_store.get_documents()
    
    def clear(self, doc_id: Optional[str] = None):
        """Delete a document or all documents."""
        if doc_id:
            self.vector_store.delete_document(doc_id)
            print(f"✓ Deleted: {doc_id}")
        else:
            # Clear all
            for doc in self.list_docs():
                self.vector_store.delete_document(doc['doc_id'])
            print("✓ Cleared all documents")

    def review(self, source: str, relay_url: str = "http://43.157.205.88:5679", 
               relay_token: str = None) -> Dict:
        """
        Send a file/code to Opus relay for review, then store the review as a doc in vault.
        Returns: {success, review_text, doc_id (of stored review)}
        """
        import json
        from pathlib import Path
        
        # Step 1: Read file content or use as-is
        source_path = Path(source)
        if source_path.exists() and source_path.is_file():
            try:
                with open(source_path, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                filename = source_path.name
            except Exception as e:
                return {"success": False, "error": f"Failed to read file: {e}"}
        else:
            code_content = source
            filename = "inline_code"
        
        # Step 2: POST to relay with Bearer auth
        try:
            headers = {
                "Content-Type": "application/json"
            }
            if relay_token:
                headers["Authorization"] = f"Bearer {relay_token}"
            
            payload = {
                "code": code_content,
                "context": f"File: {filename}"
            }
            
            response = requests.post(
                f"{relay_url.rstrip('/')}/v1/messages",
                json=payload,
                headers=headers,
                timeout=300
            )
            
            if response.status_code != 200:
                return {"success": False, "error": f"Relay returned {response.status_code}: {response.text}"}
            
            result = response.json()
            review_text = result.get("review", "")
            
        except requests.RequestException as e:
            return {"success": False, "error": f"Failed to connect to relay: {e}"}
        
        # Step 3: Ingest the review as a new document into vault
        try:
            review_doc_id = f"review_{int(time.time())}_{hash(code_content) % 1000000}"
            review_title = f"Review: {filename}"
            
            # Create a markdown document from the review
            review_markdown = f"""# {review_title}

**Source File:** {filename}
**Reviewed At:** {datetime.now().isoformat()}

## Review

{review_text}
"""
            
            # Create document object
            doc = Document(
                doc_id=review_doc_id,
                title=review_title,
                source=source,
                source_type="review",
                content=review_markdown
            )
            
            # Chunk and embed
            chunks_text = self.chunker.chunk(review_markdown)
            embeddings = self.embedding_engine.embed(chunks_text)
            chunks = []
            
            for i, (text, embedding) in enumerate(zip(chunks_text, embeddings)):
                chunk = Chunk(
                    chunk_id=f"{doc.doc_id}_chunk_{i}",
                    doc_id=doc.doc_id,
                    text=text,
                    start_pos=0,
                    end_pos=len(text),
                    embedding=embedding
                )
                chunks.append(chunk)
            
            # Store in vault
            self.vector_store.add_document(doc, chunks)
            
            return {
                "success": True,
                "review_text": review_text,
                "doc_id": review_doc_id,
                "title": review_title
            }
            
        except Exception as e:
            return {"success": False, "error": f"Failed to store review in vault: {e}"}
