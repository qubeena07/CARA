import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal
from ingestion.ingest import ingest_file


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system")
    parser.add_argument("--dir", help="Directory containing PDF/txt/md files")
    parser.add_argument("--file", help="Single file to ingest")
    args = parser.parse_args()

    if not args.file and not args.dir:
        parser.error("Provide --file or --dir")

    db = SessionLocal()

    try:
        if args.file:
            source_name = os.path.basename(args.file)
            count = ingest_file(args.file, source_name, db)
            print(f"Done. Inserted {count} chunks.")
        elif args.dir:
            total = 0
            for filename in sorted(os.listdir(args.dir)):
                if filename.endswith((".pdf", ".txt", ".md")):
                    file_path = os.path.join(args.dir, filename)
                    count = ingest_file(file_path, filename, db)
                    total += count
            print(f"\nTotal chunks inserted: {total}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
