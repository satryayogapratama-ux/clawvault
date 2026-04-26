# ClawVault: Local RAG Knowledge Base Skill

Access and query a personal knowledge base using semantic search. All processing is local — no external APIs, no privacy concerns.

## Installation

1. Ensure ClawVault is in your OpenClaw workspace:
   ```
   ~/.openclaw/workspace/clawvault/
   ```

2. Install dependencies:
   ```bash
   pip install -r clawvault/requirements.txt
   ```

## Commands

### `vault add <source>`

Ingest a document into the knowledge base.

**Formats:**
- **URL**: `vault add https://example.com/article`
- **PDF**: `vault add /path/to/document.pdf`
- **Markdown**: `vault add /path/to/notes.md`
- **Text**: `vault add /path/to/file.txt`

**Examples:**
```
vault add https://en.wikipedia.org/wiki/Climate_Change
vault add ~/Documents/research.pdf
vault add ~/notes/project-ideas.md
vault add ~/contracts/agreement.txt
```

**What happens:**
1. Document is fetched/loaded
2. Text is split into overlapping chunks (500 tokens, 50-token overlap)
3. Chunks are embedded using sentence-transformers
4. Embeddings stored in local SQLite database

### `vault ask <question>`

Query the knowledge base with a natural language question.

**Examples:**
```
vault ask What are the main causes of climate change?
vault ask How does the project handle authentication?
vault ask What are the key terms in this contract?
```

**What happens:**
1. Question is embedded using the same model
2. Semantic search finds most similar chunks (top-5 by default)
3. Retrieved chunks are returned as context
4. You can use this context to answer questions or instruct the agent

**Output:**
```
Question: What are the main causes of climate change?
Found 5 relevant chunks:

[1] Similarity: 0.842
    The primary causes of climate change include greenhouse gas emissions...

[2] Similarity: 0.791
    Carbon dioxide, methane, and other gases trap heat in the atmosphere...

[Context for further processing...]
```

### `vault list`

Show all documents currently in the knowledge base.

**Output:**
```
Documents in vault:

1. Climate_Change
   Source: https://en.wikipedia.org/wiki/Climate_Change
   Type: url
   
2. research
   Source: ~/Documents/research.pdf
   Type: pdf
```

### `vault clear [doc-id]`

Delete a document from the knowledge base.

**Examples:**
```
vault clear Climate_Change    # Delete specific document
vault clear                    # Delete all documents
```

## How It Works

### Architecture

```
Document Source (PDF, MD, URL, TXT)
         ↓
  Document Ingester
         ↓
  Text Chunker (500 tokens, overlap)
         ↓
  Embedding Engine (sentence-transformers)
         ↓
  Vector Store (SQLite + numpy)
         ↓
  [Query] → Semantic Search → Top-K Chunks → [Answer]
```

### Key Components

**DocumentIngester**
- Loads PDF (PyPDF2), Markdown, plain text, URLs (BeautifulSoup)
- Extracts and cleans content
- Handles encoding and encoding errors gracefully

**TextChunker**
- Splits by paragraphs first, respects sentence/code boundaries
- Approximate token size: 500 tokens (rough: 1 word ≈ 1.3 tokens)
- 50-token overlap for context continuity
- Prevents breaking code blocks or structured text

**EmbeddingEngine**
- Uses `sentence-transformers` (all-MiniLM-L6-v2 model)
- 384-dimensional embeddings
- Deterministic: same text always produces same embedding
- Runs entirely offline

**VectorStore**
- SQLite database (no external DB needed)
- Stores documents, chunks, and embeddings
- Cosine similarity search
- Supports document deletion (cascades to chunks)

**QueryEngine**
- Embeds query using same model
- Finds top-K similar chunks (default: 5)
- Returns chunks + formatted context string
- Can be passed to agent for answer generation

## Use Cases

### Personal Knowledge Management
```
vault add ~/Documents/notes/*.md
vault ask What was the outcome of the Q3 review meeting?
```

### Research & Reference
```
vault add https://arxiv.org/pdf/2310.00001.pdf
vault ask What are the limitations of this approach?
```

### Company Documentation
```
vault add https://internal.company.com/docs/policies.pdf
vault ask What's the PTO policy for remote workers?
```

### Contract Analysis
```
vault add ~/contracts/agreement.pdf
vault ask What are the termination conditions?
```

### Project Context
```
vault add ~/project/README.md
vault add ~/project/ARCHITECTURE.md
vault ask How does the authentication system work?
```

## Privacy & Security

- **Zero external APIs**: All processing is local
- **No data transmission**: Embeddings are computed and stored locally
- **SQLite storage**: Easy to backup, inspect, or delete
- **You control access**: Grant file system permissions as needed

## Performance

- **Embedding**: ~1-2 seconds per document (depends on size)
- **Search**: <100ms for 1000 chunks
- **Storage**: ~2MB per document (including embeddings)

Example: 10 documents = ~20MB database

## Limitations & Fallbacks

- **PDF support**: Requires PyPDF2. If unavailable, skip PDFs.
- **Network**: URLs require internet. Falls back gracefully.
- **Model loading**: If sentence-transformers unavailable, uses dummy embeddings (still functional).

All operations degrade gracefully rather than failing entirely.

## Advanced Usage

### Custom Model
```python
from clawvault import ClawVault
vault = ClawVault(model_name="all-MiniLM-L12-v2")  # Different model
```

### Direct Python API
```python
from clawvault import ClawVault

vault = ClawVault()
vault.add("~/documents/note.md")
result = vault.ask("What's the main topic?")

print(result['context'])  # Retrieved chunks formatted as context
print(result['chunks'])   # List of {text, similarity, chunk_id}
```

### Batch Operations
```python
for doc in Path("~/documents").glob("*.md"):
    vault.add(str(doc))

vault.ask("summarize the key findings")
```

## Troubleshooting

**Issue: "sentence-transformers not installed"**
- Install: `pip install sentence-transformers`

**Issue: "PyPDF2 not installed"**
- Install: `pip install PyPDF2`
- Or: Skip PDFs, use other formats

**Issue: Large documents are slow to embed**
- Normal for first run (model downloads ~60MB)
- Subsequent runs use cached model
- For very large documents, split before ingesting

**Issue: Search returns irrelevant results**
- Try rephrasing your question (semantic search works best with natural language)
- Increase top_k in vault.ask() if needed

## Model Details

**Model**: `all-MiniLM-L6-v2`
- Lightweight: ~80MB
- Fast: embeds 1000 sentences in ~2 seconds
- Accurate: Competitive with larger models for retrieval tasks
- Multilingual support: Works reasonably well in 50+ languages

## License

ClawVault is provided under the Proprietary Evaluation License. See LICENSE file for details.

## Contributing

Found a bug or want to suggest a feature? Please report it to the maintainers.
