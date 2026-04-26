#!/usr/bin/env python3
"""
ClawVault Demo: RAG System in Action

Demonstrates:
1. Loading documents from URL and markdown
2. Chunking and embedding
3. Semantic search
4. Retrieved context for Q&A
"""

import sys
import tempfile
from pathlib import Path
from clawvault import ClawVault


def create_sample_markdown() -> str:
    """Create a sample markdown file about OpenClaw."""
    content = """# OpenClaw: Your Personal AI Agent

## What is OpenClaw?

OpenClaw is a personal AI assistant that runs locally on your machine. It integrates with 
your files, calendar, messages, and devices to help you work smarter and faster.

## Key Features

- **Local-first**: Your data stays on your machine. No cloud, no third-party access.
- **Skill-based**: Extensible architecture lets you add custom skills and integrations.
- **Multi-channel**: Works with Telegram, Discord, Slack, and other messaging platforms.
- **Hands-free**: Control via voice, browser, or keyboard shortcuts.

## Architecture

OpenClaw runs as a daemon on your machine, connecting to:
- Your file system
- Smart home devices (cameras, screens, sensors)
- Messaging apps
- Browser extension for web context

## Getting Started

1. Install OpenClaw: `npm install -g openclaw`
2. Run the gateway: `openclaw gateway start`
3. Connect to your preferred messaging platform
4. Start asking questions!

## Use Cases

- **Knowledge Management**: ClawVault RAG for personal documents
- **Task Automation**: Custom skills for repetitive workflows
- **Device Control**: Smart home integration
- **Information Retrieval**: Semantic search across your files
- **Content Creation**: AI-assisted writing and research

## Privacy & Security

All processing happens locally. No model weights are transmitted. You control what data 
the agent can access through the security policy.

## Future Roadmap

- Improved memory management
- Multi-agent collaboration
- Advanced scheduling
- Plugin marketplace
"""
    return content


def main():
    print("=" * 60)
    print("ClawVault Demo: Local RAG Knowledge Base")
    print("=" * 60)
    print()
    
    # Initialize vault in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "demo.db"
        print(f"Creating vault at: {db_path}")
        vault = ClawVault(str(db_path))
        print()
        
        # 1. Ingest Wikipedia URL
        print("Step 1: Ingesting documents")
        print("-" * 60)
        
        wikipedia_url = "https://en.wikipedia.org/wiki/Indonesia"
        success = vault.add(wikipedia_url, source_type='url')
        
        if not success:
            print(f"WARNING: Could not fetch {wikipedia_url}")
            print("This is normal if offline. Continuing with local demo...")
        
        print()
        
        # 2. Create and ingest sample markdown
        sample_md = create_sample_markdown()
        sample_path = Path(tmpdir) / "openclaw.md"
        sample_path.write_text(sample_md)
        
        vault.add(str(sample_path), source_type='markdown')
        print()
        
        # 3. List documents
        print("Step 2: Documents in vault")
        print("-" * 60)
        docs = vault.list_docs()
        if docs:
            for doc in docs:
                print(f"  • {doc['title']}")
                print(f"    Source: {doc['source']}")
                print(f"    Type: {doc['source_type']}")
                print()
        else:
            print("No documents ingested. (Network may be unavailable)")
            print()
        
        # 4. Ask questions
        print("Step 3: Semantic search")
        print("-" * 60)
        
        questions = [
            "What are the key features of OpenClaw?",
            "How does privacy work in OpenClaw?",
            "What platforms does OpenClaw support?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}: {question}")
            print("-" * 40)
            
            result = vault.ask(question, top_k=3)
            
            if result['chunks']:
                print(f"Found {len(result['chunks'])} relevant chunks:\n")
                
                for j, chunk in enumerate(result['chunks'], 1):
                    similarity = chunk['similarity']
                    text = chunk['text'][:150] + "..." if len(chunk['text']) > 150 else chunk['text']
                    
                    print(f"  [{j}] Similarity: {similarity:.3f}")
                    print(f"      {text}")
                    print()
            else:
                print("No relevant chunks found.")
                print("(This may happen if documents didn't load due to network issues.)")
            
            print()
        
        print("=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Try vault.add('path/to/document.pdf') for PDFs")
        print("  2. Try vault.add('path/to/notes.md') for markdown")
        print("  3. Use vault.ask('your question') for semantic search")
        print("  4. vault.list_docs() to see all documents")
        print("  5. vault.clear(doc_id) to remove a document")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
