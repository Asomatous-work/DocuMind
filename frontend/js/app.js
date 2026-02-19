/**
 * DocuMind — AI Document Intelligence
 * Main Application JavaScript
 * 
 * Handles: Upload, Camera Capture, OCR Processing, Chat, Knowledge Base
 */

// ═══════════════════════════════════════════════════════════════════════════
// API HELPER
// ═══════════════════════════════════════════════════════════════════════════

const API = {
    BASE: '',

    async post(endpoint, body, isFormData = false) {
        const options = {
            method: 'POST',
            body: isFormData ? body : JSON.stringify(body),
        };
        if (!isFormData) {
            options.headers = { 'Content-Type': 'application/json' };
        }
        const res = await fetch(`${this.BASE}${endpoint}`, options);
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Request failed');
        }
        return res.json();
    },

    async get(endpoint) {
        const res = await fetch(`${this.BASE}${endpoint}`);
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Request failed');
        }
        return res.json();
    },

    async del(endpoint) {
        const res = await fetch(`${this.BASE}${endpoint}`, { method: 'DELETE' });
        if (!res.ok) throw new Error('Delete failed');
        return res.json();
    },
};

// ═══════════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════════

const state = {
    documents: [],
    isProcessing: false,
    isChatting: false,
    cameraStream: null,
    welcomeVisible: true,
};

// ═══════════════════════════════════════════════════════════════════════════
// DOM REFERENCES
// ═══════════════════════════════════════════════════════════════════════════

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    sidebar: $('#sidebar'),
    sidebarToggle: $('#sidebarToggle'),
    uploadZone: $('#uploadZone'),
    fileInput: $('#fileInput'),
    captureBtn: $('#captureBtn'),
    uploadBtn: $('#uploadBtn'),
    documentsList: $('#documentsList'),
    emptyDocs: $('#emptyDocs'),
    docCount: $('#docCount'),
    statBlocks: $('#statBlocks'),
    statConfidence: $('#statConfidence'),
    messagesContainer: $('#messagesContainer'),
    messagesInner: $('#messagesInner'),
    welcomeScreen: $('#welcomeScreen'),
    chatInput: $('#chatInput'),
    sendBtn: $('#sendBtn'),
    clearChatBtn: $('#clearChatBtn'),
    healthBtn: $('#healthBtn'),
    statusDot: $('#statusDot'),
    cameraModal: $('#cameraModal'),
    cameraVideo: $('#cameraVideo'),
    cameraCanvas: $('#cameraCanvas'),
    closeCameraBtn: $('#closeCameraBtn'),
    snapBtn: $('#snapBtn'),
    docDetailModal: $('#docDetailModal'),
    docDetailContent: $('#docDetailContent'),
    toastContainer: $('#toastContainer'),
};

// ═══════════════════════════════════════════════════════════════════════════
// TOAST NOTIFICATIONS
// ═══════════════════════════════════════════════════════════════════════════

function showToast(message, type = 'info', duration = 4000) {
    const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${icons[type] || ''}</span><span>${message}</span>`;
    dom.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ═══════════════════════════════════════════════════════════════════════════
// UPLOAD & DRAG-DROP
// ═══════════════════════════════════════════════════════════════════════════

function initUpload() {
    // Click on zone or Upload button
    dom.uploadZone.addEventListener('click', () => dom.fileInput.click());
    dom.uploadBtn.addEventListener('click', () => dom.fileInput.click());

    // File selected
    dom.fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 0) processFiles(files);
        dom.fileInput.value = ''; // Reset so same file can be re-uploaded
    });

    // Drag & Drop
    dom.uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dom.uploadZone.classList.add('drag-over');
    });

    dom.uploadZone.addEventListener('dragleave', () => {
        dom.uploadZone.classList.remove('drag-over');
    });

    dom.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dom.uploadZone.classList.remove('drag-over');
        const files = Array.from(e.dataTransfer.files).filter((f) =>
            f.type.startsWith('image/')
        );
        if (files.length > 0) {
            processFiles(files);
        } else {
            showToast('Please drop image files only', 'warning');
        }
    });
}

async function processFiles(files) {
    if (state.isProcessing) {
        showToast('Already processing a document, please wait...', 'warning');
        return;
    }

    for (const file of files) {
        await processFile(file);
    }
}

async function processFile(file) {
    state.isProcessing = true;
    hideWelcome();

    // Show processing card in chat
    const processingId = addProcessingCard(file.name);

    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('source_type', 'upload');

        const result = await API.post('/api/ocr/upload', formData, true);

        // Remove processing card
        removeProcessingCard(processingId);

        if (result.success) {
            // Show OCR result in chat
            addOCRResultMessage(result.document);
            showToast(`"${file.name}" processed successfully!`, 'success');
            await refreshDocuments();
        }
    } catch (err) {
        removeProcessingCard(processingId);
        addSystemMessage(`❌ Failed to process "${file.name}": ${err.message}`);
        showToast(`Error: ${err.message}`, 'error');
    }

    state.isProcessing = false;
}

// ═══════════════════════════════════════════════════════════════════════════
// CAMERA CAPTURE
// ═══════════════════════════════════════════════════════════════════════════

function initCamera() {
    dom.captureBtn.addEventListener('click', openCamera);
    dom.closeCameraBtn.addEventListener('click', closeCamera);
    dom.snapBtn.addEventListener('click', captureSnapshot);

    // Close on background click
    dom.cameraModal.addEventListener('click', (e) => {
        if (e.target === dom.cameraModal) closeCamera();
    });
}

async function openCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment', // Prefer rear camera on mobile
                width: { ideal: 1920 },
                height: { ideal: 1080 },
            },
        });
        state.cameraStream = stream;
        dom.cameraVideo.srcObject = stream;
        dom.cameraModal.classList.add('active');
    } catch (err) {
        showToast('Camera access denied or not available', 'error');
        console.error('Camera error:', err);
    }
}

function closeCamera() {
    if (state.cameraStream) {
        state.cameraStream.getTracks().forEach((t) => t.stop());
        state.cameraStream = null;
    }
    dom.cameraVideo.srcObject = null;
    dom.cameraModal.classList.remove('active');
}

async function captureSnapshot() {
    if (!state.cameraStream) return;

    const video = dom.cameraVideo;
    const canvas = dom.cameraCanvas;

    // Set canvas to video dimensions for max quality
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    // Flash effect
    const flash = document.createElement('div');
    flash.style.cssText = `
    position: absolute; inset: 0; background: white;
    opacity: 0.6; pointer-events: none; z-index: 10;
    animation: flashAnim 0.3s ease forwards;
  `;
    const style = document.createElement('style');
    style.textContent = `@keyframes flashAnim { to { opacity: 0; } }`;
    document.head.appendChild(style);
    dom.cameraVideo.parentElement.appendChild(flash);
    setTimeout(() => {
        flash.remove();
        style.remove();
    }, 400);

    // Get base64 image
    const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
    closeCamera();

    // Process the capture
    hideWelcome();
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const filename = `capture_${timestamp}.jpg`;

    const processingId = addProcessingCard(filename);
    state.isProcessing = true;

    try {
        const result = await API.post('/api/ocr/capture', {
            image_base64: dataUrl,
            filename: filename,
        });

        removeProcessingCard(processingId);

        if (result.success) {
            addOCRResultMessage(result.document);
            showToast('Document captured and processed!', 'success');
            await refreshDocuments();
        }
    } catch (err) {
        removeProcessingCard(processingId);
        addSystemMessage(`❌ Capture processing failed: ${err.message}`);
        showToast(`Error: ${err.message}`, 'error');
    }

    state.isProcessing = false;
}

// ═══════════════════════════════════════════════════════════════════════════
// CHAT
// ═══════════════════════════════════════════════════════════════════════════

function initChat() {
    // Auto-resize textarea
    dom.chatInput.addEventListener('input', () => {
        dom.chatInput.style.height = 'auto';
        dom.chatInput.style.height = Math.min(dom.chatInput.scrollHeight, 120) + 'px';
        dom.sendBtn.disabled = !dom.chatInput.value.trim();
    });

    // Send on Enter (Shift+Enter for newline)
    dom.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    dom.sendBtn.addEventListener('click', sendMessage);

    // Clear chat
    dom.clearChatBtn.addEventListener('click', async () => {
        dom.messagesInner.innerHTML = '';
        state.welcomeVisible = true;
        showWelcome();
        try {
            await API.post('/api/chat/clear');
            showToast('Conversation cleared', 'info');
        } catch (_) { }
    });

    // Health check
    dom.healthBtn.addEventListener('click', checkHealth);
}

async function sendMessage() {
    const text = dom.chatInput.value.trim();
    if (!text || state.isChatting) return;

    state.isChatting = true;
    hideWelcome();
    addUserMessage(text);

    dom.chatInput.value = '';
    dom.chatInput.style.height = 'auto';
    dom.sendBtn.disabled = true;

    // Show typing indicator
    const typingEl = addTypingIndicator();

    try {
        const result = await API.post('/api/chat', {
            message: text,
            use_knowledge: true,
        });

        removeElement(typingEl);
        addAssistantMessage(result.response, result.sources);
    } catch (err) {
        removeElement(typingEl);
        addAssistantMessage(`⚠️ Error: ${err.message}`);
    }

    state.isChatting = false;
    dom.chatInput.focus();
}

// ═══════════════════════════════════════════════════════════════════════════
// MESSAGE RENDERING
// ═══════════════════════════════════════════════════════════════════════════

function getTimeString() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function addUserMessage(text) {
    const html = `
    <div class="message user">
      <div class="message-avatar">👤</div>
      <div class="message-content">
        <div class="message-bubble">${escapeHtml(text)}</div>
        <div class="message-time">${getTimeString()}</div>
      </div>
    </div>
  `;
    dom.messagesInner.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
}

function addAssistantMessage(text, sources = []) {
    const formattedText = formatMarkdown(text);
    let sourcesHtml = '';
    if (sources.length > 0) {
        sourcesHtml = `
      <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border-subtle);">
        <span style="font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">Sources:</span>
        ${sources.map((s) => `<span style="font-size: 11px; color: var(--text-accent); margin-left: 6px;">📄 ${escapeHtml(s.filename)}</span>`).join('')}
      </div>
    `;
    }

    const html = `
    <div class="message assistant">
      <div class="message-avatar">🧠</div>
      <div class="message-content">
        <div class="message-bubble">${formattedText}${sourcesHtml}</div>
        <div class="message-time">${getTimeString()}</div>
      </div>
    </div>
  `;
    dom.messagesInner.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
}

function addSystemMessage(text) {
    const html = `
    <div class="message assistant">
      <div class="message-avatar">⚙️</div>
      <div class="message-content">
        <div class="message-bubble">${text}</div>
        <div class="message-time">${getTimeString()}</div>
      </div>
    </div>
  `;
    dom.messagesInner.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
}

function addOCRResultMessage(doc) {
    const confidenceClass =
        doc.avg_confidence >= 0.85 ? 'confidence-high' :
            doc.avg_confidence >= 0.6 ? 'confidence-medium' : 'confidence-low';

    const textPreview = doc.extracted_text
        ? doc.extracted_text.substring(0, 500) + (doc.extracted_text.length > 500 ? '...' : '')
        : '(No text detected)';

    const html = `
    <div class="ocr-result-card">
      <div class="ocr-result-header">
        <span class="ocr-icon">📝</span>
        <div class="ocr-title">
          <h4>Text Extracted Successfully</h4>
          <span>${escapeHtml(doc.filename)}</span>
        </div>
      </div>
      <div class="ocr-stats">
        <div class="ocr-stat">
          <span class="stat-label">Blocks:</span>
          <span class="stat-val">${doc.block_count}</span>
        </div>
        <div class="ocr-stat">
          <span class="stat-label">Confidence:</span>
          <span class="stat-val ${confidenceClass}">${(doc.avg_confidence * 100).toFixed(1)}%</span>
        </div>
        <div class="ocr-stat">
          <span class="stat-label">Time:</span>
          <span class="stat-val">${doc.processing_time}s</span>
        </div>
      </div>
      <div class="ocr-text-preview">${escapeHtml(textPreview)}</div>
    </div>
  `;
    dom.messagesInner.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
}

function addProcessingCard(filename) {
    const id = 'proc-' + Date.now();
    const html = `
    <div class="processing-card" id="${id}">
      <div class="processing-header">
        <div class="processing-spinner"></div>
        <h4>Processing "${escapeHtml(filename)}"...</h4>
      </div>
      <div class="processing-progress"><div class="bar"></div></div>
    </div>
  `;
    dom.messagesInner.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
    return id;
}

function removeProcessingCard(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function addTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'typing-indicator';
    div.innerHTML = `
    <div class="message-avatar" style="background: var(--accent-gradient); box-shadow: 0 0 12px var(--accent-glow);">🧠</div>
    <div class="typing-dots">
      <span></span><span></span><span></span>
    </div>
  `;
    dom.messagesInner.appendChild(div);
    scrollToBottom();
    return div;
}

function removeElement(el) {
    if (el && el.parentNode) el.parentNode.removeChild(el);
}

// ═══════════════════════════════════════════════════════════════════════════
// DOCUMENTS LIST
// ═══════════════════════════════════════════════════════════════════════════

async function refreshDocuments() {
    try {
        const data = await API.get('/api/documents');
        state.documents = data.documents || [];
        renderDocuments();
        updateStats(data.stats);
    } catch (err) {
        console.error('Failed to refresh documents:', err);
    }
}

function renderDocuments() {
    if (state.documents.length === 0) {
        dom.emptyDocs.classList.remove('hidden');
        dom.docCount.textContent = '0 docs';
        // Remove any doc-item elements but keep empty-docs
        dom.documentsList.querySelectorAll('.doc-item').forEach((el) => el.remove());
        return;
    }

    dom.emptyDocs.classList.add('hidden');
    dom.docCount.textContent = `${state.documents.length} doc${state.documents.length !== 1 ? 's' : ''}`;

    // Remove existing doc items
    dom.documentsList.querySelectorAll('.doc-item').forEach((el) => el.remove());

    // Render each document
    state.documents.forEach((doc) => {
        const sourceIcon =
            doc.source_type === 'camera' ? '📸' :
                doc.source_type === 'digital' ? '🖥️' : '📄';

        const confidenceClass =
            doc.ocr_confidence >= 0.85 ? 'confidence-high' :
                doc.ocr_confidence >= 0.6 ? 'confidence-medium' : 'confidence-low';

        const timeAgo = formatTimeAgo(doc.created_at);

        const html = `
      <div class="doc-item" data-doc-id="${doc.id}">
        <div class="doc-icon">${sourceIcon}</div>
        <div class="doc-info">
          <div class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
          <div class="doc-meta">
            <span>${timeAgo}</span>
            <span class="confidence-badge ${confidenceClass}">${(doc.ocr_confidence * 100).toFixed(0)}%</span>
          </div>
        </div>
        <button class="doc-delete" title="Delete" data-delete-id="${doc.id}">🗑️</button>
      </div>
    `;
        dom.documentsList.insertAdjacentHTML('beforeend', html);
    });

    // Bind click events
    dom.documentsList.querySelectorAll('.doc-item').forEach((el) => {
        el.addEventListener('click', (e) => {
            if (e.target.closest('.doc-delete')) return;
            const docId = el.dataset.docId;
            showDocDetail(docId);
        });
    });

    dom.documentsList.querySelectorAll('.doc-delete').forEach((btn) => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const docId = btn.dataset.deleteId;
            if (confirm('Delete this document from the knowledge base?')) {
                try {
                    await API.del(`/api/documents/${docId}`);
                    showToast('Document deleted', 'info');
                    await refreshDocuments();
                } catch (err) {
                    showToast('Failed to delete document', 'error');
                }
            }
        });
    });
}

function updateStats(stats) {
    if (!stats) return;
    dom.statBlocks.textContent = stats.total_blocks || 0;
    dom.statConfidence.textContent =
        stats.avg_confidence > 0
            ? (stats.avg_confidence * 100).toFixed(0) + '%'
            : '—';
}

async function showDocDetail(docId) {
    try {
        const doc = await API.get(`/api/documents/${docId}`);
        const confidenceClass =
            doc.ocr_confidence >= 0.85 ? 'confidence-high' :
                doc.ocr_confidence >= 0.6 ? 'confidence-medium' : 'confidence-low';

        dom.docDetailContent.innerHTML = `
      <div class="camera-modal-header">
        <h3>📄 ${escapeHtml(doc.filename)}</h3>
        <button class="close-modal-btn" id="closeDocDetailBtn">✕</button>
      </div>
      <div class="ocr-stats" style="margin-bottom: 16px;">
        <div class="ocr-stat">
          <span class="stat-label">Source:</span>
          <span class="stat-val">${doc.source_type}</span>
        </div>
        <div class="ocr-stat">
          <span class="stat-label">Blocks:</span>
          <span class="stat-val">${doc.block_count}</span>
        </div>
        <div class="ocr-stat">
          <span class="stat-label">Confidence:</span>
          <span class="stat-val ${confidenceClass}">${(doc.ocr_confidence * 100).toFixed(1)}%</span>
        </div>
      </div>
      <h4 style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 8px;">Extracted Text</h4>
      <div class="ocr-text-preview" style="max-height: 400px;">${escapeHtml(doc.extracted_text || '(No text)')}</div>
      <div style="margin-top: 12px; font-size: 11px; color: var(--text-muted);">
        Created: ${new Date(doc.created_at).toLocaleString()}
      </div>
    `;

        dom.docDetailModal.classList.add('active');

        document.getElementById('closeDocDetailBtn').addEventListener('click', () => {
            dom.docDetailModal.classList.remove('active');
        });
    } catch (err) {
        showToast('Could not load document details', 'error');
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HEALTH CHECK
// ═══════════════════════════════════════════════════════════════════════════

async function checkHealth() {
    try {
        const health = await API.get('/api/health');
        const ollamaStatus = health.ollama === 'connected';
        dom.statusDot.classList.toggle('offline', !ollamaStatus);

        addSystemMessage(
            `<strong>System Status</strong><br/>` +
            `• Ollama: ${ollamaStatus ? '🟢 Connected' : '🔴 Disconnected'}<br/>` +
            `• Model: <code>${escapeHtml(health.model)}</code><br/>` +
            `• Documents: ${health.knowledge_base?.total_documents || 0}<br/>` +
            `• Total blocks: ${health.knowledge_base?.total_blocks || 0}`
        );
        hideWelcome();
    } catch (err) {
        dom.statusDot.classList.add('offline');
        showToast('Cannot reach backend server', 'error');
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SIDEBAR TOGGLE (Mobile)
// ═══════════════════════════════════════════════════════════════════════════

function initSidebar() {
    dom.sidebarToggle.addEventListener('click', () => {
        dom.sidebar.classList.toggle('open');
    });

    // Close sidebar on outside click (mobile)
    document.addEventListener('click', (e) => {
        if (
            dom.sidebar.classList.contains('open') &&
            !dom.sidebar.contains(e.target) &&
            e.target !== dom.sidebarToggle
        ) {
            dom.sidebar.classList.remove('open');
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// WELCOME SCREEN
// ═══════════════════════════════════════════════════════════════════════════

function hideWelcome() {
    if (state.welcomeVisible && dom.welcomeScreen) {
        dom.welcomeScreen.classList.add('hidden');
        state.welcomeVisible = false;
    }
}

function showWelcome() {
    if (dom.welcomeScreen) {
        dom.welcomeScreen.classList.remove('hidden');
        state.welcomeVisible = true;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatMarkdown(text) {
    if (!text) return '';
    // Simple markdown-like formatting
    let html = escapeHtml(text);
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italic
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Inline code
    html = html.replace(/`(.*?)`/g, '<code style="background:var(--bg-glass);padding:2px 6px;border-radius:4px;font-family:var(--font-mono);font-size:12px;">$1</code>');
    // Line breaks
    html = html.replace(/\n/g, '<br/>');
    // Bullet points
    html = html.replace(/^- (.*)/gm, '• $1');
    return html;
}

function formatTimeAgo(dateStr) {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHrs = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHrs < 24) return `${diffHrs}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        dom.messagesContainer.scrollTop = dom.messagesContainer.scrollHeight;
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// CLOSE MODALS
// ═══════════════════════════════════════════════════════════════════════════

// Close doc detail on background click
dom.docDetailModal.addEventListener('click', (e) => {
    if (e.target === dom.docDetailModal) {
        dom.docDetailModal.classList.remove('active');
    }
});

// Escape key to close modals
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        if (dom.cameraModal.classList.contains('active')) closeCamera();
        if (dom.docDetailModal.classList.contains('active'))
            dom.docDetailModal.classList.remove('active');
    }
});

// ═══════════════════════════════════════════════════════════════════════════
// INITIALIZE
// ═══════════════════════════════════════════════════════════════════════════

async function init() {
    initUpload();
    initCamera();
    initChat();
    initSidebar();

    // Load existing documents
    await refreshDocuments();

    // Check health in background
    try {
        const health = await API.get('/api/health');
        dom.statusDot.classList.toggle('offline', health.ollama !== 'connected');
    } catch (_) {
        dom.statusDot.classList.add('offline');
    }

    console.log('🔍 DocuMind initialized');
}

// Wait for DOM
document.addEventListener('DOMContentLoaded', init);
