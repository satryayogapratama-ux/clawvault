#!/usr/bin/env python3
"""
vault_search.py — Simple CLI for searching ClawVault documents.
Usage:
    python3 vault_search.py "xero chart of accounts"
    python3 vault_search.py "BPJS rates" --top 3
    python3 vault_search.py --list                   # list all ingested docs
    python3 vault_search.py --ingest path/to/file.pdf
"""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from clawvault import DocumentIngester, TextChunker, EmbeddingEngine, VectorStore, Chunk, QueryEngine

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


def main():
    parser = argparse.ArgumentParser(description="ClawVault document search")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--top", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--list", action="store_true", help="List all ingested documents")
    parser.add_argument("--ingest", metavar="FILE", help="Ingest a new document")
    args = parser.parse_args()

    if args.list:
        list_docs()
    elif args.ingest:
        ingest_file(args.ingest)
    elif args.query:
        search(args.query, top_k=args.top)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
