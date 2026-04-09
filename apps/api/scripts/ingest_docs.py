"""
CLI script to ingest documents into the vector store.
Usage: python scripts/ingest_docs.py --path /path/to/docs
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ingestion.ingest import ingest_document


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Cara vector store")
    parser.add_argument("--path", required=True, help="Path to file or directory")
    args = parser.parse_args()

    path = args.path
    if os.path.isfile(path):
        with open(path, "r") as f:
            content = f.read()
        records = ingest_document(content, source=path)
        print(f"Ingested {len(records)} chunks from {path}")
    elif os.path.isdir(path):
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            if os.path.isfile(fpath):
                with open(fpath, "r") as f:
                    content = f.read()
                records = ingest_document(content, source=fpath)
                print(f"Ingested {len(records)} chunks from {fpath}")
    else:
        print(f"Path not found: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
