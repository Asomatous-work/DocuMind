import json
import os

file_path = "/Users/aravindg/demo-OCR/backend/knowledge/data/documents.json"

def clean_duplicates():
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    documents = data.get("documents", [])
    
    # Sort documents so that those WITH checksums come first
    # This ensures we keep the one with metadata if there's a duplicate
    documents.sort(key=lambda x: (x.get("checksum") is None, x.get("created_at", "")))

    unique_docs = []
    seen_checksums = set()
    seen_filenames = set()

    for doc in documents:
        checksum = doc.get("checksum")
        filename = doc.get("filename")
        
        if checksum:
            if checksum not in seen_checksums:
                seen_checksums.add(checksum)
                unique_docs.append(doc)
                if filename:
                    seen_filenames.add(filename)
            else:
                print(f"Skipping duplicate by checksum: {filename} ({checksum})")
        else:
            # For documents without checksum
            if filename not in seen_filenames:
                seen_filenames.add(filename)
                unique_docs.append(doc)
                print(f"Added non-checksum doc: {filename}")
            else:
                print(f"Skipping duplicate by filename (no checksum): {filename}")

    data["documents"] = unique_docs
    data["metadata"]["total_documents"] = len(unique_docs)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Cleaned documents.json. Total unique docs: {len(unique_docs)}")

if __name__ == "__main__":
    clean_duplicates()
