"""
Demo-OCR API Server
FastAPI backend powering the OCR Chat Interface.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import logging
import httpx
import json
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)-20s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo-ocr")

OLLAMA_URL = "http://127.0.0.1:11434"

# Late imports to avoid slow startup logs before config
from ocr.engine import OCREngine
from ocr.document_extractor import (
    extract_document,
    DOCUMENT_MIME_TYPES,
    DOCUMENT_EXTENSIONS,
)
from knowledge.store import KnowledgeStore
from agent.ollama_client import OllamaAgent

# ─── Initialize App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Demo-OCR",
    description="AI-powered OCR Chat Interface with local LLM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Services ─────────────────────────────────────────────────────────────────

ocr_engine = OCREngine()
knowledge_store = KnowledgeStore()
ollama_agent = OllamaAgent()

# ─── Serve Frontend ──────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")
app.mount(
    "/assets",
    StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")),
    name="assets",
)


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main chat interface."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# ─── Pydantic Models ─────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    use_knowledge: bool = True


class CaptureRequest(BaseModel):
    image_base64: str
    filename: str = "camera_capture.jpg"


class ChatResponse(BaseModel):
    response: str
    sources: list = []


# ─── API Routes ───────────────────────────────────────────────────────────────


@app.post("/api/ocr/upload")
async def upload_and_process(
    file: UploadFile = File(...),
    source_type: str = Form("upload"),
):
    """
    Upload a document image, PDF, or Word document and extract text.
    Images are processed via OCR; PDFs and DOCX files use direct text extraction.
    Stores the result in the knowledge base.
    """
    ext = os.path.splitext((file.filename or "").lower())[1]

    # Determine if this is a native document or image
    is_document = (
        file.content_type in DOCUMENT_MIME_TYPES
        or ext in DOCUMENT_EXTENSIONS
    )

    # Image MIME types (OCR path)
    image_types = [
        "image/jpeg", "image/png", "image/webp", "image/bmp",
        "image/tiff", "image/gif",
    ]
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
    is_image = file.content_type in image_types or ext in image_extensions

    if not is_document and not is_image:
        allowed = "Images (JPG, PNG, WebP, BMP, TIFF), PDF, DOCX"
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type or ext}. Allowed: {allowed}",
        )

    # Read file bytes
    file_bytes = await file.read()
    file_size = len(file_bytes)
    
    # Calculate checksum for duplicate detection
    checksum = hashlib.sha256(file_bytes).hexdigest()
    
    # Check if this exact file already exists
    existing_doc = knowledge_store.find_by_checksum(checksum)
    if existing_doc:
        logger.info(f"♻️ Duplicate file detected: {file.filename} (ID: {existing_doc['id']})")
        return {
            "success": True,
            "duplicate": True,
            "document": {
                "id": existing_doc["id"],
                "filename": existing_doc["filename"],
                "extracted_text": existing_doc["extracted_text"],
                "block_count": existing_doc["block_count"],
                "avg_confidence": existing_doc["ocr_confidence"],
                "processing_time": 0,
                "created_at": existing_doc["created_at"],
            },
        }

    logger.info(f"📤 Processing upload: {file.filename} ({file_size} bytes)")

    try:
        if is_document:
            # Direct text extraction — no OCR
            ocr_result = extract_document(
                file_bytes,
                filename=file.filename or "unknown",
                mime_type=file.content_type or "",
            )
        else:
            # Image — run through OCR engine
            # Optimization: If it's a screenshot, use the light digital pipeline
            current_source = source_type
            if "screenshot" in (file.filename or "").lower():
                current_source = "digital"
                
            ocr_result = ocr_engine.extract_text(
                file_bytes, source_type=current_source, detail=True
            )
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

    # Store in knowledge base
    document = knowledge_store.add_document(
        filename=file.filename or "unknown",
        extracted_text=ocr_result["text"],
        source_type=ocr_result.get("source_type", source_type),
        ocr_confidence=ocr_result["avg_confidence"],
        ocr_blocks=ocr_result["blocks"],
        file_size=file_size,
        mime_type=file.content_type or "",
        checksum=checksum,
    )

    return {
        "success": True,
        "document": {
            "id": document["id"],
            "filename": document["filename"],
            "extracted_text": ocr_result["text"],
            "block_count": ocr_result["block_count"],
            "avg_confidence": ocr_result["avg_confidence"],
            "processing_time": ocr_result["processing_time_seconds"],
            "image_url": "", # No longer storing images
        },
    }


@app.post("/api/ocr/capture")
async def capture_and_process(request: CaptureRequest):
    """
    Process a camera-captured image (sent as base64).
    """
    logger.info(f"📸 Processing camera capture: {request.filename}")

    # Read base64
    import base64
    try:
        header, data = request.image_base64.split(",", 1)
        image_bytes = base64.b64decode(data)
    except Exception as e:
        logger.error(f"Failed to decode base64: {e}")
        image_bytes = b""

    # Check for duplicates even in camera captures
    checksum = hashlib.sha256(image_bytes).hexdigest()
    existing_doc = knowledge_store.find_by_checksum(checksum)
    if existing_doc:
        logger.info(f"♻️ Duplicate capture detected: {request.filename} (ID: {existing_doc['id']})")
        return {
            "success": True,
            "duplicate": True,
            "document": {
                "id": existing_doc["id"],
                "filename": existing_doc["filename"],
                "extracted_text": existing_doc["extracted_text"],
                "block_count": existing_doc["block_count"],
                "avg_confidence": existing_doc["ocr_confidence"],
                "processing_time": 0,
            },
        }

    try:
        ocr_result = ocr_engine.extract_from_base64(
            request.image_base64, source_type="camera", detail=True
        )
    except Exception as e:
        logger.error(f"Camera OCR failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

    # Store in knowledge base
    document = knowledge_store.add_document(
        filename=request.filename,
        extracted_text=ocr_result["text"],
        source_type="camera",
        ocr_confidence=ocr_result["avg_confidence"],
        ocr_blocks=ocr_result["blocks"],
        checksum=checksum,
    )

    return {
        "success": True,
        "document": {
            "id": document["id"],
            "filename": document["filename"],
            "extracted_text": ocr_result["text"],
            "block_count": ocr_result["block_count"],
            "avg_confidence": ocr_result["avg_confidence"],
            "processing_time": ocr_result["processing_time_seconds"],
            "image_url": "", # No longer storing images
        },
    }


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat response from the AI agent.
    """
    context = ""
    sources = []
    
    if request.use_knowledge:
        context = knowledge_store.get_context_for_query(request.message)
        if context:
            search_results = knowledge_store.search(request.message)
            sources = [
                {"filename": r["filename"], "id": r["id"]}
                for r in search_results
            ]

    def generate():
        # First yield the sources as a standard JSON line
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"
        
        response_gen = ollama_agent.chat(
            user_message=request.message,
            document_context=context,
            stream=True
        )
        for chunk in response_gen:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield json.dumps({"type": "content", "content": content}) + "\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the AI agent (blocking).
    """
    context = ""
    sources = []

    if request.use_knowledge:
        context = knowledge_store.get_context_for_query(request.message)
        if context:
            search_results = knowledge_store.search(request.message)
            sources = [
                {"filename": r["filename"], "id": r["id"]}
                for r in search_results
            ]

    response = ollama_agent.chat(
        user_message=request.message,
        document_context=context,
    )

    return ChatResponse(response=response, sources=sources)


@app.get("/api/documents")
async def list_documents():
    """Get all documents in the knowledge base."""
    docs = knowledge_store.get_all_documents()
    stats = knowledge_store.get_stats()
    return {"documents": docs, "stats": stats}


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get a specific document by ID."""
    doc = knowledge_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the knowledge base."""
    success = knowledge_store.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"success": True, "message": f"Document {doc_id} deleted"}


@app.get("/api/knowledge/stats")
async def knowledge_stats():
    """Get knowledge base statistics."""
    return knowledge_store.get_stats()


@app.post("/api/chat/clear")
async def clear_chat():
    """Clear conversation history."""
    ollama_agent.clear_history()
    return {"success": True, "message": "Conversation history cleared"}


@app.get("/api/ollama/status")
async def ollama_status():
    """Direct check of Ollama service status."""
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            return {"status": "connected", "details": r.json()}
        except Exception as e:
            return {"status": "disconnected", "error": str(e)}


@app.get("/api/health")
async def health():
    """Health check and system status."""
    ollama_connected = False
    models = []
    
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            ollama_connected = True
            models = r.json().get('models', [])
        except:
            pass

    return {
        "status": "healthy",
        "ollama": "connected" if ollama_connected else "disconnected",
        "model": ollama_agent.model,
        "available_models": [m.get('name') for m in models],
        "knowledge_base": knowledge_store.get_stats(),
    }
