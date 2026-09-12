/**
 * Personal AI Assistant — Frontend Application Logic
 * ChatGPT-style UI with dark/light theme toggle.
 */

// ═══════════════════════════════════════════════════════════════
// DOM References
// ═══════════════════════════════════════════════════════════════
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const chatMessages    = $('#chat-messages');
const messageInput    = $('#message-input');
const sendBtn         = $('#send-btn');
const typingIndicator = $('#typing-indicator');
const welcomeContainer = $('#welcome-container');
const currentModeBadge = $('#current-mode');

const sidebar      = $('#sidebar');
const menuBtn      = $('#menu-btn');
const sidebarClose = $('#sidebar-close');

const uploadZone    = $('#upload-zone');
const pdfInput      = $('#pdf-input');
const uploadProgress = $('#upload-progress');
const progressFill  = $('#progress-fill');
const progressText  = $('#progress-text');
const pdfList       = $('#pdf-list');

const notesPanel    = $('#notes-panel');
const btnToggleNotes = $('#btn-toggle-notes');
const notesClose    = $('#notes-close');
const noteInput     = $('#note-input');
const noteSaveBtn   = $('#note-save-btn');
const notesList     = $('#notes-list');

const btnNewChat    = $('#btn-new-chat');
const btnClearChat  = $('#btn-clear-chat');
const themeToggle   = $('#theme-toggle');
const toastContainer = $('#toast-container');

// ═══════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════
let isProcessing = false;
let sidebarOverlay = null;

/** Generate a UUID v4 for session isolation. */
function generateSessionId() {
    return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
        (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
    );
}

// Unique ID for this chat session. Regenerated on every "New Chat".
let sessionId = generateSessionId();

// ═══════════════════════════════════════════════════════════════
// Initialize
// ═══════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initEventListeners();
    loadUploadedPDFs();
    autoResizeTextarea(messageInput);
});

// ═══════════════════════════════════════════════════════════════
// Theme
// ═══════════════════════════════════════════════════════════════
function initTheme() {
    const saved = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
}

// ═══════════════════════════════════════════════════════════════
// Event Listeners
// ═══════════════════════════════════════════════════════════════
function initEventListeners() {
    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    messageInput.addEventListener('input', () => {
        sendBtn.disabled = !messageInput.value.trim();
        autoResizeTextarea(messageInput);
    });

    // Welcome chips
    $$('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            messageInput.value = chip.dataset.message;
            sendBtn.disabled = false;
            sendMessage();
        });
    });

    // Sidebar
    menuBtn.addEventListener('click', openSidebar);
    sidebarClose.addEventListener('click', closeSidebar);

    // Upload
    uploadZone.addEventListener('click', () => pdfInput.click());
    pdfInput.addEventListener('change', handleFileUpload);
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) uploadFile(files[0]);
    });

    // Notes
    btnToggleNotes.addEventListener('click', () => {
        notesPanel.classList.toggle('open');
        if (notesPanel.classList.contains('open')) loadNotes();
    });
    notesClose.addEventListener('click', () => notesPanel.classList.remove('open'));
    noteSaveBtn.addEventListener('click', saveNote);

    // Clear chat
    btnNewChat.addEventListener('click', clearChat);
    btnClearChat.addEventListener('click', clearChat);

    // Theme toggle
    themeToggle.addEventListener('click', toggleTheme);
}

// ═══════════════════════════════════════════════════════════════
// Chat
// ═══════════════════════════════════════════════════════════════
let currentAbortController = null;
const stopBtn = $('#stop-btn');

if (stopBtn) {
    stopBtn.addEventListener('click', () => {
        if (currentAbortController) {
            currentAbortController.abort();
        }
    });
}

async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || isProcessing) return;

    isProcessing = true;

    if (welcomeContainer) welcomeContainer.style.display = 'none';

    appendMessage('user', text);
    messageInput.value = '';
    
    sendBtn.disabled = true;
    sendBtn.style.display = 'none';
    stopBtn.style.display = 'flex';
    
    autoResizeTextarea(messageInput);
    showTyping();

    currentAbortController = new AbortController();
    let botBubble = null;
    let fullResponse = "";

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, session_id: sessionId }),
            signal: currentAbortController.signal
        });

        hideTyping();

        if (!res.ok) {
            if (res.status === 429) {
                appendRateLimitMessage();
            } else {
                appendMessage('bot', '⚠️ Oops! Something went wrong on my end. Please try again.', 'GENERAL');
                showToast('error', 'Failed to get response');
            }
            throw new Error("HTTP Error " + res.status);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const chunkText = decoder.decode(value, { stream: true });
            const lines = chunkText.split('\n\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.substring(6));
                        if (data.type === 'category') {
                            updateModeBadge(data.category);
                            if (!botBubble) botBubble = createBotMessageShell(data.category);
                        } else if (data.type === 'chunk') {
                            if (!botBubble) botBubble = createBotMessageShell('GENERAL');
                            fullResponse += data.text;
                            botBubble.innerHTML = formatBotResponse(fullResponse);
                            requestAnimationFrame(() => {
                                chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'auto' });
                            });
                        } else if (data.type === 'error') {
                            if (!botBubble) botBubble = createBotMessageShell('GENERAL');
                            botBubble.innerHTML = formatBotResponse(data.text);
                        }
                    } catch (e) {
                        console.error('Error parsing SSE json', e, line);
                    }
                }
            }
        }

    } catch (err) {
        hideTyping();
        if (err.name === 'AbortError') {
             showToast('info', 'Generation stopped');
        } else if (!err.message.startsWith('HTTP')) {
            appendMessage('bot', '⚠️ Could not reach the server. Please check your connection and try again.', 'GENERAL');
            showToast('error', 'Connection failed');
        }
    } finally {
        isProcessing = false;
        sendBtn.style.display = 'flex';
        sendBtn.disabled = !messageInput.value.trim();
        stopBtn.style.display = 'none';
        currentAbortController = null;
        messageInput.focus();
    }
}

function createBotMessageShell(category) {
    const wrapper = document.createElement('div');
    wrapper.classList.add('message', 'bot');

    const avatar = document.createElement('div');
    avatar.classList.add('message-avatar');
    avatar.textContent = 'AI';

    const body = document.createElement('div');
    body.classList.add('message-body');

    if (category) {
        const catBadge = document.createElement('div');
        catBadge.classList.add('message-category', `cat-${category}`);
        catBadge.textContent = category;
        body.appendChild(catBadge);
    }

    const bubble = document.createElement('div');
    bubble.classList.add('message-content');
    
    body.appendChild(bubble);
    wrapper.appendChild(avatar);
    wrapper.appendChild(body);
    chatMessages.appendChild(wrapper);
    
    return bubble;
}

function appendMessage(role, content, category = null) {
    const wrapper = document.createElement('div');
    wrapper.classList.add('message', role);

    const avatar = document.createElement('div');
    avatar.classList.add('message-avatar');
    avatar.textContent = role === 'user' ? 'You' : 'AI';

    const body = document.createElement('div');
    body.classList.add('message-body');

    if (role === 'bot' && category) {
        const catBadge = document.createElement('div');
        catBadge.classList.add('message-category', `cat-${category}`);
        catBadge.textContent = category;
        body.appendChild(catBadge);
    }

    const bubble = document.createElement('div');
    bubble.classList.add('message-content');

    if (role === 'bot') {
        bubble.innerHTML = formatBotResponse(content);
    } else {
        bubble.textContent = content;
    }

    body.appendChild(bubble);
    wrapper.appendChild(avatar);
    wrapper.appendChild(body);
    chatMessages.appendChild(wrapper);

    requestAnimationFrame(() => {
        chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
    });
}

function appendRateLimitMessage() {
    const wrapper = document.createElement('div');
    wrapper.classList.add('message', 'bot');

    const avatar = document.createElement('div');
    avatar.classList.add('message-avatar');
    avatar.textContent = 'AI';

    const body = document.createElement('div');
    body.classList.add('message-body');

    const catBadge = document.createElement('div');
    catBadge.classList.add('message-category', 'cat-GENERAL');
    catBadge.textContent = 'RATE LIMIT';
    body.appendChild(catBadge);

    const bubble = document.createElement('div');
    bubble.classList.add('message-content', 'rate-limit-bubble');
    bubble.innerHTML = `
        <div class="rate-limit-icon">⏳</div>
        <p><strong>API quota reached</strong></p>
        <p>I've used up my free-tier API calls for the moment. Please wait <strong>30–60 seconds</strong> and try again.</p>
        <p style="font-size:0.8em; opacity:0.7; margin-top:6px;">Tip: You can upgrade your <a href="https://ai.google.dev/pricing" target="_blank" style="color:inherit;">Gemini API plan</a> for a higher quota.</p>
    `;

    body.appendChild(bubble);
    wrapper.appendChild(avatar);
    wrapper.appendChild(body);
    chatMessages.appendChild(wrapper);

    showToast('error', '⏳ Rate limit hit — wait 30–60s then retry');

    requestAnimationFrame(() => {
        chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
    });
}

function formatBotResponse(text) {
    if (!text) return '';

    let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Code blocks
    html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Lists
    html = html.replace(/^[\-\*]\s+(.+)/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
    html = html.replace(/^\d+\.\s+(.+)/gm, '<li>$1</li>');

    // Paragraphs
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');

    if (!html.startsWith('<')) {
        html = `<p>${html}</p>`;
    }

    return html;
}

function updateModeBadge(category) {
    currentModeBadge.textContent = category;
    currentModeBadge.setAttribute('data-mode', category);
}

function showTyping() {
    typingIndicator.style.display = 'block';
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
}

function hideTyping() {
    typingIndicator.style.display = 'none';
}

async function clearChat() {
    // Tell the server to wipe memory for the current session
    try {
        await fetch('/api/new-chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });
    } catch (_) { /* silently ignore network errors during reset */ }

    // Generate a fresh session ID so this chat is fully isolated
    sessionId = generateSessionId();

    chatMessages.querySelectorAll('.message').forEach(m => m.remove());
    if (welcomeContainer) welcomeContainer.style.display = 'flex';
    updateModeBadge('GENERAL');
    showToast('info', 'New chat started');
}

// ═══════════════════════════════════════════════════════════════
// PDF Upload
// ═══════════════════════════════════════════════════════════════
function handleFileUpload(e) {
    const file = e.target.files[0];
    if (file) uploadFile(file);
}

async function uploadFile(file) {
    if (!file.name.endsWith('.pdf')) {
        showToast('error', 'Only PDF files are supported');
        return;
    }

    uploadProgress.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = `Uploading ${file.name}...`;

    let progress = 0;
    const progressInterval = setInterval(() => {
        progress = Math.min(progress + Math.random() * 15, 85);
        progressFill.style.width = `${progress}%`;
    }, 300);

    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('session_id', sessionId);

        const res = await fetch('/api/upload-pdf', {
            method: 'POST',
            body: formData
        });

        clearInterval(progressInterval);

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Upload failed');
        }

        const data = await res.json();
        progressFill.style.width = '100%';
        progressText.textContent = 'Done!';

        setTimeout(() => {
            uploadProgress.style.display = 'none';
        }, 1800);

        showToast('success', data.message);
        loadUploadedPDFs();

    } catch (err) {
        clearInterval(progressInterval);
        uploadProgress.style.display = 'none';
        if (err.message && (err.message.includes('quota') || err.message.includes('rate') || err.message.includes('429'))) {
            showToast('error', '⏳ API quota reached — wait 30–60s and retry');
        } else {
            showToast('error', err.message);
        }
    }

    pdfInput.value = '';
}

async function loadUploadedPDFs() {
    try {
        const res = await fetch('/api/uploaded-pdfs');
        if (!res.ok) return;

        const data = await res.json();
        pdfList.innerHTML = '';

        if (data.pdfs.length === 0) {
            pdfList.innerHTML = '<div style="font-size:0.78rem;opacity:0.5;padding:6px 0;">No PDFs uploaded yet.</div>';
            return;
        }

        data.pdfs.forEach(pdf => {
            const item = document.createElement('div');
            item.classList.add('pdf-item');
            item.innerHTML = `
                <span class="pdf-item-name" title="${pdf}">${pdf}</span>
                <button class="pdf-delete-btn" title="Remove ${pdf}" onclick="deletePdf('${pdf.replace(/'/g, "\\'")}', this)">🗑️</button>
            `;
            pdfList.appendChild(item);
        });

    } catch (err) {
        // Silently fail
    }
}

async function deletePdf(filename, btnEl) {
    if (!confirm(`Remove "${filename}" from the knowledge base?`)) return;
    btnEl.disabled = true;
    btnEl.textContent = '⏳';

    try {
        const res = await fetch(`/api/delete-pdf?filename=${encodeURIComponent(filename)}&session_id=${sessionId}`, {
            method: 'DELETE'
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Delete failed');
        showToast('success', `"${filename}" removed!`);
        loadUploadedPDFs();
    } catch (err) {
        btnEl.disabled = false;
        btnEl.textContent = '🗑️';
        showToast('error', err.message);
    }
}

// ═══════════════════════════════════════════════════════════════
// Notes
// ═══════════════════════════════════════════════════════════════
async function loadNotes() {
    try {
        const res = await fetch('/api/notes');
        if (!res.ok) return;
        const data = await res.json();
        renderNotes(data.notes);
    } catch (err) {
        notesList.innerHTML = '<div class="notes-empty"><p>Failed to load notes.</p></div>';
    }
}

function renderNotes(notesText) {
    if (!notesText || notesText === 'No personal notes found.') {
        notesList.innerHTML = '<div class="notes-empty"><p>No notes yet.</p></div>';
        return;
    }

    const lines = notesText.split('\n');
    notesList.innerHTML = '';

    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;

        const match = line.match(/^\d+\.\s*\[([^\]]+)\]\s*(.*)/);
        if (match) {
            const item = document.createElement('div');
            item.classList.add('note-item');
            item.innerHTML = `
                <div class="note-item-time">${match[1]}</div>
                <div class="note-item-content">${match[2]}</div>
            `;
            notesList.appendChild(item);
        }
    }

    if (notesList.children.length === 0) {
        notesList.innerHTML = '<div class="notes-empty"><p>No notes yet.</p></div>';
    }
}

async function saveNote() {
    const content = noteInput.value.trim();
    if (!content) return;

    try {
        const res = await fetch('/api/notes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Failed to save note');
        }

        noteInput.value = '';
        showToast('success', 'Note saved');
        loadNotes();

    } catch (err) {
        showToast('error', err.message);
    }
}

// ═══════════════════════════════════════════════════════════════
// Sidebar (Mobile)
// ═══════════════════════════════════════════════════════════════
function openSidebar() {
    sidebar.classList.add('open');
    if (!sidebarOverlay) {
        sidebarOverlay = document.createElement('div');
        sidebarOverlay.classList.add('sidebar-overlay');
        sidebarOverlay.addEventListener('click', closeSidebar);
        document.body.appendChild(sidebarOverlay);
    }
    sidebarOverlay.classList.add('active');
}

function closeSidebar() {
    sidebar.classList.remove('open');
    if (sidebarOverlay) sidebarOverlay.classList.remove('active');
}

// ═══════════════════════════════════════════════════════════════
// Toast Notifications
// ═══════════════════════════════════════════════════════════════
function showToast(type, message) {
    const toast = document.createElement('div');
    toast.classList.add('toast', `toast-${type}`);

    const icons = { success: '✓', error: '✕', info: 'i' };
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || 'i'}</span>
        <span>${message}</span>
    `;

    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-out');
        setTimeout(() => toast.remove(), 300);
    }, 3200);
}

// ═══════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════
function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
}
