#!/usr/bin/env python3
"""
vault_search.py — Simple CLI for searching ClawVault documents.
Usage:
    python3 vault_search.py "xero chart of accounts"
    python3 vault_search.py "BPJS rates" --top 3
    python3 vault_search.py --list                   # list all ingested docs
    python3 vault_search.py --ingest path/to/file.pdf
    python3 vault_search.py review path/to/file.py  # send to Opus relay for review
    python3 vault_search.py review path/to/file.py --relay-url http://localhost:5679 --relay-token TOKEN
"""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from clawvault import DocumentIngester, TextChunker, EmbeddingEngine, VectorStore, Chunk, QueryEngine, ClawVault

DB_PATH = str(Path(__file__).parent / "clawvault.db")


def search(query: str, top_k: int = 5):
    engine = EmbeddingEngine()
    store = VectorStore(DB_PATH)
    qe = QueryEngine(engine, store)
    result = qe.query(query, top_k=top_k)

    print(f"\nQuery: {query}\n{'─'*60}")
    chunks = result.get("chunks", [])
    if not chunks:
        print("No results found.")
        return

    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("similarity", chunk.get("score", 0))
        text = chunk.get("text", "")[:400]
        doc_id = chunk.get("chunk_id", "")
        print(f"\n[{i}] score={score:.3f}")
        print(text)
        if len(chunk.get("text", "")) > 400:
            print("... [truncated]")
    print()


def list_docs():
    store = VectorStore(DB_PATH)
    docs = store.get_documents()
    if not docs:
        print("No documents ingested yet.")
        return
    print(f"\n{'─'*60}")
    print(f"{'Title':30} {'Type':10} {'Ingested'}")
    print(f"{'─'*60}")
    for d in docs:
        print(f"{d['title'][:30]:30} {d['source_type']:10} {d.get('ingested_at','')[:10]}")
    print(f"\nTotal: {len(docs)} documents")


def ingest_file(path: str):
    engine = EmbeddingEngine()
    store = VectorStore(DB_PATH)
    chunker = TextChunker(chunk_size=200, overlap=20)

    doc = DocumentIngester.ingest(path)
    if not doc:
        print(f"❌ Failed to ingest: {path}")
        return

    chunks_text = chunker.chunk(doc.content)
    chunks = []
    for i, ct in enumerate(chunks_text):
        cid = hashlib.md5(f"{doc.doc_id}_{i}".encode()).hexdigest()[:12]
        emb = engine.embed([ct])
        chunks.append(Chunk(chunk_id=cid, doc_id=doc.doc_id, text=ct,
                           start_pos=i*200, end_pos=(i+1)*200, embedding=emb[0]))

    store.add_document(doc, chunks)
    print(f"✅ Ingested: {doc.title} ({len(chunks)} chunks)")


def review_file(filepath: str, relay_url: str = "http://43.157.205.88:5679", relay_token: str = None):
    """Send a file to Opus relay for review and store result in vault."""
    vault = ClawVault(db_path=DB_PATH)
    result = vault.review(filepath, relay_url=relay_url, relay_token=relay_token)
    
    if not result.get("success"):
        print(f"❌ Review failed: {result.get('error')}")
        return
    
    print(f"✅ Review complete: {result.get('title')}")
    print(f"   Doc ID: {result.get('doc_id')}")
    print(f"\n📋 Review:\n{result.get('review_text')[:500]}...")


def main():
    parser = argparse.ArgumentParser(description="ClawVault document search & review")
    parser.add_argument("query", nargs="?", help="Search query or 'review' for code review")
    parser.add_argument("filepath", nargs="?", help="File path (for review command)")
    parser.add_argument("--top", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--list", action="store_true", help="List all ingested documents")
    parser.add_argument("--ingest", metavar="FILE", help="Ingest a new document")
    parser.add_argument("--relay-url", default="http://43.157.205.88:5679", help="Opus relay URL")
    parser.add_argument("--relay-token", help="Opus relay auth token")
    args = parser.parse_args()

    if args.list:
        list_docs()
    elif args.ingest:
        ingest_file(args.ingest)
    elif args.query == "review":
        if not args.filepath:
            print("Error: file path required for review command")
            print("Usage: python3 vault_search.py review <filepath> [--relay-url URL] [--relay-token TOKEN]")
            sys.exit(1)
        review_file(args.filepath, relay_url=args.relay_url, relay_token=args.relay_token)
    elif args.query:
        search(args.query, top_k=args.top)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
