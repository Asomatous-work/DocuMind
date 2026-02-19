# DocuMind — AI Document Intelligence

> AI-powered OCR Chat Interface with local LLM intelligence.  
> Scan, capture, and understand any document.

## 🏗️ Architecture

```
demo-OCR/
├── backend/
│   ├── main.py                  # FastAPI server
│   ├── requirements.txt         # Python dependencies
│   ├── ocr/
│   │   ├── engine.py            # EasyOCR wrapper (high-accuracy)
│   │   └── preprocessor.py      # CV preprocessing pipeline
│   ├── knowledge/
│   │   ├── store.py             # JSON knowledge base
│   │   └── data/
│   │       └── documents.json   # Document storage
│   └── agent/
│       └── ollama_client.py     # Ollama LLM integration
├── frontend/
│   ├── index.html               # Main chat interface
│   ├── css/
│   │   └── style.css            # Glassmorphic design system
│   ├── js/
│   │   └── app.js               # Application logic
│   └── assets/                  # Static assets
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

### 1. Install Dependencies

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull Ollama Model

```bash
ollama pull tinyllama
```

### 3. Run the Server

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open in Browser

Navigate to [http://localhost:8000](http://localhost:8000)

## 📸 Features

| Feature | Description |
| :--- | :--- |
| **Document Upload** | Drag & drop or browse for images (JPG, PNG, WebP, BMP, TIFF) |
| **Camera Capture** | Real-time document scanning using device camera |
| **High-Accuracy OCR** | EasyOCR with 7-stage preprocessing: resize → deskew → shadow removal → denoise → CLAHE → sharpen |
| **Knowledge Base** | JSON storage with keyword search — no vector DB needed |
| **AI Chat** | Local Ollama LLM answers questions about your documents |
| **Premium UI** | Dark glassmorphic design with micro-animations |

## 🔧 OCR Preprocessing Pipeline

1. **Resize** — Upscale small images to 300 DPI equivalent
2. **Deskew** — Detect and correct document rotation
3. **Shadow Removal** — Remove lighting artifacts
4. **Denoise** — Non-local means denoising
5. **CLAHE** — Adaptive contrast enhancement
6. **Sharpen** — Edge enhancement for text

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/ocr/upload` | Upload and OCR a document |
| `POST` | `/api/ocr/capture` | Process camera capture (base64) |
| `POST` | `/api/chat` | Chat with AI about documents |
| `GET` | `/api/documents` | List all documents |
| `GET` | `/api/documents/:id` | Get document details |
| `DELETE` | `/api/documents/:id` | Delete a document |
| `GET` | `/api/health` | System health check |
