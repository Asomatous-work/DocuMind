"""
JSON Knowledge Base Store
Simple, file-based document storage for OCR-extracted text.
No vector DB needed — uses keyword search for retrieval.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional
import logging

from ocr.text_cleaner import clean_ocr_text, chunk_ocr_text

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_FILE = os.path.join(DATA_DIR, "documents.json")


class KnowledgeStore:
    """
    JSON-based knowledge base for storing and retrieving OCR documents.
    Optimized for tiny LLMs by using structured chunking and section-aware search.
    """

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(DB_FILE):
            self._init_db()
        self._load()

    def _init_db(self):
        """Initialize empty database file."""
        data = {
            "documents": [],
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "version": "1.1",
                "total_documents": 0,
            },
        }
        self._save(data)

    def _load(self) -> dict:
        """Load the JSON database into memory."""
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("⚠️ Corrupted or missing DB, reinitializing...")
            self._init_db()
            with open(DB_FILE, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        return self._data

    def _save(self, data: Optional[dict] = None):
        """Write the current state to disk."""
        if data is None:
            data = self._data
        data["metadata"]["total_documents"] = len(data["documents"])
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_document(
        self,
        filename: str,
        extracted_text: str,
        source_type: str,
        ocr_confidence: float,
        ocr_blocks: list,
        file_size: int = 0,
        mime_type: str = "",
        checksum: str = "",
    ) -> dict:
        """
        Store a new OCR-processed document in the knowledge base.
        Automatically cleans and chunks the text for better LLM grounding.
        """
        doc_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        # Clean and chunk the text immediately
        cleaned_text = clean_ocr_text(extracted_text)
        chunks = chunk_ocr_text(extracted_text)

        document = {
            "id": doc_id,
            "filename": filename,
            "extracted_text": cleaned_text,
            "chunks": chunks,
            "source_type": source_type,
            "ocr_confidence": ocr_confidence,
            "block_count": len(ocr_blocks),
            "blocks": [],  # Clear blocks for main storage to reduce noise/size
            "file_size_bytes": file_size,
            "mime_type": mime_type,
            "created_at": now,
            "updated_at": now,
            "checksum": checksum,
            "tags": [],
            "notes": "",
        }

        self._data["documents"].append(document)
        self._save()

        # Optionally save full block data to a sidecar file if needed
        # (For now we omit it to keep the DB file clean as requested)

        logger.info(f"📄 Document stored & chunked: {doc_id} — {filename}")
        return document

    def get_all_documents(self) -> list:
        """Get all documents (metadata only)."""
        self._load()
        docs = []
        for doc in self._data["documents"]:
            summary = {
                "id": doc["id"],
                "filename": doc["filename"],
                "source_type": doc["source_type"],
                "ocr_confidence": doc["ocr_confidence"],
                "chunk_count": len(doc.get("chunks", [])),
                "created_at": doc["created_at"],
                "text_preview": doc["extracted_text"][:200] + "...",
            }
            docs.append(summary)
        return docs

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Get a specific document by ID."""
        self._load()
        for doc in self._data["documents"]:
            if doc["id"] == doc_id:
                return doc
        return None

    def find_by_checksum(self, checksum: str) -> Optional[dict]:
        """Find a document by its file checksum."""
        if not checksum:
            return None
        self._load()
        for doc in self._data["documents"]:
            if doc.get("checksum") == checksum:
                return doc
        return None

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        self._load()
        original_len = len(self._data["documents"])
        self._data["documents"] = [
            d for d in self._data["documents"] if d["id"] != doc_id
        ]
        if len(self._data["documents"]) < original_len:
            self._save()
            logger.info(f"🗑️ Document deleted: {doc_id}")
            return True
        return False

    def search(self, query: str, top_k: int = 5) -> list:
        """
        Search across documents. Uses FUZZY matching for section labels and keywords.
        """
        self._load()
        from rapidfuzz import fuzz, process
        
        query_lower = query.lower()
        
        # 1. Extract potential section markers (e.g., s12, sz12, s 12)
        import re
        potential_nums = re.findall(r'[sz]?\s*(\d+)', query_lower)
        requested_labels = [f"S{num}" for num in potential_nums]

        # 2. Extract potential keywords (filter out short noise)
        query_words = [w for w in query_lower.split() if len(w) > 3]

        scored_docs = []
        for doc in self._data["documents"]:
            text = doc["extracted_text"]
            text_lower = text.lower()
            score = 0
            
            # --- Keyword Fuzzy Matching ---
            if query_lower in text_lower:
                score += 100
            
            # Additional word-based scoring
            for word in query_words:
                if word in text_lower:
                    score += 40
                else:
                    # Fuzzy match for word
                    # partial_ratio is good for "securit" -> "security"
                    best_word_score = fuzz.partial_ratio(word, text_lower)
                    if best_word_score > 85:
                        score += 30

            # --- Section Label Fuzzy Matching ---
            chunk_labels = [c["label"] for c in doc.get("chunks", [])]
            if chunk_labels and requested_labels:
                for req in requested_labels:
                    match = process.extractOne(req, chunk_labels, scorer=fuzz.ratio)
                    if match and match[1] > 85:
                        score += 150
            
            if score > 0:
                scored_docs.append({
                    "id": doc["id"],
                    "filename": doc["filename"],
                    "score": score,
                    "text": text,
                    "chunks": doc.get("chunks", []),
                    "created_at": doc["created_at"],
                })

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]

    @staticmethod
    def extract_relevant_snippet(doc: dict, query: str) -> str:
        """
        Search for the most relevant chunk(s) within a document for a given query.
        """
        from rapidfuzz import fuzz, process
        query_lower = query.lower()
        
        # 1. Keywords to focus on (ignore noise)
        query_words = [w for w in query_lower.split() if len(w) > 3]
        if not query_words:
            # Fallback for very short queries
            return doc.get("extracted_text", "")[:1200]

        chunks = doc.get("chunks", [])
        if not chunks:
            # Document hasn't been chunked yet or failed?
            return doc.get("extracted_text", "")[:1200]

        # 2. Score each chunk based on keywords
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            text_lower = chunk["text"].lower()
            score = 0
            for word in query_words:
                if word in text_lower:
                    score += 50
                else:
                    # Fuzzy word search within chunk
                    # better than searching entire doc
                    match_score = fuzz.partial_ratio(word, text_lower)
                    if match_score > 85:
                        score += 30
            
            if score > 0:
                chunk_scores.append((score, i))

        # 3. Handle matches
        if not chunk_scores:
            # No specific chunk matched well, return beginning or random sample?
            return (f"({doc['filename']} - Overview): " + 
                   doc.get("extracted_text", "")[:1000])

        # Sort by score and pick top 1-2 chunks
        chunk_scores.sort(key=lambda x: x[0], reverse=True)
        
        selected_parts = []
        # Return up to 2 high-scoring chunks
        for score, idx in chunk_scores[:2]:
            chunk = chunks[idx]
            selected_parts.append(f"[{chunk['label']}]: {chunk['text']}")
            
        return "\n\n".join(selected_parts)

    def get_context_for_query(self, query: str, max_chars: int = 2500) -> str:
        """
        Aggregates relevant chunks across the best matching documents.
        Handles both section specific queries (S1, S2) and general keyword searches.
        """
        results = self.search(query, top_k=3)
        if not results:
            return ""

        from rapidfuzz import fuzz, process
        import re
        query_lower = query.lower()
        
        # 1. Look for explicit section markers in query (S1, S2, etc)
        requested_nums = re.findall(r'[sz]?\s*(\d+)', query_lower)
        requested_labels = [f"S{num}" for num in requested_nums]
        
        context_parts = []
        found_sections = set()

        # 2. Process Section matches first if requested
        if requested_labels:
            for res in results:
                chunks = {c["label"]: c["text"] for c in res.get("chunks", [])}
                chunk_labels = list(chunks.keys())
                
                for req in requested_labels:
                    # Fuzzy match requested label against this document's chunks
                    match = process.extractOne(req, chunk_labels, scorer=fuzz.ratio)
                    if match and match[1] > 85:
                        matched_label = match[0]
                        context_parts.append(f"[From: {res['filename']} - {matched_label}]\n{chunks[matched_label]}")
                        found_sections.add(req)

        # 3. Always add general keyword-based matches (crucial for names like 'Prakash')
        # This will either find chunks the section logic missed, or provide general context
        for res in results:
            snippet = self.extract_relevant_snippet(res, query)
            # Avoid duplicating if snippet is just a section we already found
            if snippet not in [p for p in context_parts]:
                context_parts.append(f"[From: {res['filename']}]\n{snippet}")

        # Join and truncate to stay within LLM limits
        full_context = "\n\n---\n\n".join(context_parts)
        if len(full_context) > max_chars:
            full_context = full_context[:max_chars] + "... [Context truncated for length]"
            
        return full_context

    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        self._load()
        docs = self._data["documents"]
        total_text = sum(len(d["extracted_text"]) for d in docs)
        return {
            "total_documents": len(docs),
            "total_chunks": sum(len(d.get("chunks", [])) for d in docs),
            "total_characters": total_text,
            "avg_confidence": (
                sum(d["ocr_confidence"] for d in docs) / len(docs)
                if docs else 0
            ),
        }
