"""
Demo-OCR API Server
FastAPI backend powering the OCR Chat Interface.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import logging
import httpx

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
    with open(index_path, "r") as f:
        return HTMLResponse(content=f.read())


# ─── Storage Configuration ──────────────────────────────────────────────────

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
IMAGES_DIR = os.path.join(STORAGE_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Mount storage directory to serve images
app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")

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
    Upload a document image and extract text via OCR.
    Stores the result in the knowledge base.
    """
    # Validate file type
    allowed_types = [
        "image/jpeg", "image/png", "image/webp", "image/bmp",
        "image/tiff", "image/gif",
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: {', '.join(allowed_types)}",
        )

    # Read file bytes
    file_bytes = await file.read()
    file_size = len(file_bytes)

    # Save image to storage for preview
    import uuid
    image_ext = file.filename.split(".")[-1] if "." in file.filename else "png"
    image_filename = f"{uuid.uuid4().hex}.{image_ext}"
    image_path = os.path.join(IMAGES_DIR, image_filename)
    
    with open(image_path, "wb") as f:
        f.write(file_bytes)

    logger.info(f"📤 Processing upload: {file.filename} ({file_size} bytes)")

    # Run OCR
    try:
        ocr_result = ocr_engine.extract_text(
            file_bytes, source_type=source_type, detail=True
        )
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

    # Store in knowledge base
    document = knowledge_store.add_document(
        filename=file.filename or "unknown",
        extracted_text=ocr_result["text"],
        source_type=source_type,
        ocr_confidence=ocr_result["avg_confidence"],
        ocr_blocks=ocr_result["blocks"],
        file_size=file_size,
        mime_type=file.content_type or "",
        image_path=f"/storage/images/{image_filename}"
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
            "image_url": document["image_path"],
        },
    }


@app.post("/api/ocr/capture")
async def capture_and_process(request: CaptureRequest):
    """
    Process a camera-captured image (sent as base64).
    """
    logger.info(f"📸 Processing camera capture: {request.filename}")

    # Save to storage
    import base64
    import uuid
    try:
        header, data = request.image_base64.split(",", 1)
        image_bytes = base64.b64decode(data)
        
        image_filename = f"{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(IMAGES_DIR, image_filename)
        with open(image_path, "wb") as f:
            f.write(image_bytes)
    except Exception as e:
        logger.error(f"Failed to save captured image: {e}")
        image_bytes = b""
        image_filename = ""

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
        image_path=f"/storage/images/{image_filename}" if image_filename else ""
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
            "image_url": document["image_path"],
        },
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the AI agent. Optionally uses knowledge base context.
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
