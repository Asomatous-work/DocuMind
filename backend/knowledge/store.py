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

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_FILE = os.path.join(DATA_DIR, "documents.json")


class KnowledgeStore:
    """
    JSON-based knowledge base for storing and retrieving OCR documents.
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
                "version": "1.0",
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
    ) -> dict:
        """
        Store a new OCR-processed document in the knowledge base.

        Returns the created document record.
        """
        doc_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        document = {
            "id": doc_id,
            "filename": filename,
            "extracted_text": extracted_text,
            "source_type": source_type,
            "ocr_confidence": ocr_confidence,
            "block_count": len(ocr_blocks),
            "blocks": ocr_blocks,
            "file_size_bytes": file_size,
            "mime_type": mime_type,
            "created_at": now,
            "updated_at": now,
            "tags": [],
            "notes": "",
        }

        self._data["documents"].append(document)
        self._save()

        logger.info(f"📄 Document stored: {doc_id} — {filename}")
        return document

    def get_all_documents(self) -> list:
        """Get all documents (metadata only, without full blocks)."""
        self._load()
        docs = []
        for doc in self._data["documents"]:
            summary = {
                "id": doc["id"],
                "filename": doc["filename"],
                "source_type": doc["source_type"],
                "ocr_confidence": doc["ocr_confidence"],
                "block_count": doc["block_count"],
                "created_at": doc["created_at"],
                "text_preview": doc["extracted_text"][:200] + "..."
                if len(doc["extracted_text"]) > 200
                else doc["extracted_text"],
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
        Keyword search across all documents.
        Returns the top_k most relevant documents with targeted snippets.
        """
        self._load()
        query_lower = query.lower()
        query_words = [w for w in query_lower.split() if len(w) > 1]

        scored_docs = []
        for doc in self._data["documents"]:
            text = doc["extracted_text"]
            text_lower = text.lower()
            score = 0
            for word in query_words:
                score += text_lower.count(word)
            # Heavy boost for exact phrase
            if query_lower in text_lower:
                score += 20

            if score > 0:
                scored_docs.append({
                    "id": doc["id"],
                    "filename": doc["filename"],
                    "score": score,
                    "text": text,
                    "created_at": doc["created_at"],
                })

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]

    @staticmethod
    def extract_relevant_snippet(text: str, query: str, window: int = 600) -> str:
        """
        Extract the most relevant snippet from a document for a given query.

        Strategy (in priority order):
        1. If query looks like a section reference (S8, S10 etc.), extract that exact section.
        2. Otherwise find the keyword and extract a tight, centered window around it.
        """
        if not text:
            return text

        query_lower = query.lower().strip()
        text_lower = text.lower()

        # --- Strategy 1: Section-aware extraction ---
        # Detect if the query is asking about a specific section like "S8", "s10", etc.
        import re
        section_match = re.search(r's(\d+)', query_lower)
        if section_match:
            section_num = section_match.group(1)
            section_label = f"S{section_num}."
            
            # Find this section in the text
            sec_idx = text.find(section_label)
            if sec_idx != -1:
                # Find the end of this section (next section start or end of text)
                next_section = re.search(r'\nS\d+\.', text[sec_idx + len(section_label):])
                if next_section:
                    sec_end = sec_idx + len(section_label) + next_section.start()
                else:
                    sec_end = len(text)
                
                snippet = text[sec_idx:sec_end].strip()
                return snippet

        # --- Strategy 2: Keyword-centered window ---
        # Find the best keyword match position
        idx = text_lower.find(query_lower)

        if idx == -1:
            # Try individual words, longest first
            words = sorted(query_lower.split(), key=len, reverse=True)
            for word in words:
                if len(word) < 2:
                    continue
                idx = text_lower.find(word)
                if idx != -1:
                    break

        if idx == -1:
            return text[:window]

        # Center the window around the match
        half = window // 2
        start = max(0, idx - half)
        end = min(len(text), idx + half)

        # Snap to paragraph boundaries (double newlines) if possible
        para_start = text.rfind('\n\n', max(0, start - 100), idx)
        if para_start != -1:
            start = para_start + 2

        para_end = text.find('\n\n', idx)
        if para_end != -1 and para_end < end + 200:
            end = para_end

        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet

    def get_context_for_query(self, query: str, max_chars: int = 2000) -> str:
        """
        Get a TARGETED snippet from the knowledge base for a given query.
        Returns the exact passage most relevant to the question,
        NOT the whole document — this prevents LLM hallucination.
        """
        results = self.search(query, top_k=2)
        if not results:
            return ""

        context_parts = []
        total_chars = 0
        for result in results:
            snippet = self.extract_relevant_snippet(result["text"], query, window=800)
            if total_chars + len(snippet) > max_chars:
                snippet = snippet[: max_chars - total_chars]
            context_parts.append(
                f"[From: {result['filename']}]\n{snippet}"
            )
            total_chars += len(snippet)
            if total_chars >= max_chars:
                break

        return "\n\n---\n\n".join(context_parts)

    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        self._load()
        docs = self._data["documents"]
        total_text = sum(len(d["extracted_text"]) for d in docs)
        return {
            "total_documents": len(docs),
            "total_characters": total_text,
            "total_blocks": sum(d["block_count"] for d in docs),
            "avg_confidence": (
                sum(d["ocr_confidence"] for d in docs) / len(docs)
                if docs
                else 0
            ),
        }
