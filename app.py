"""
FastAPI Web Server for Personal AI Chatbot.
Wraps existing core logic (chains, memory, notes) as REST APIs
and serves the premium web UI from static/ folder.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
import shutil
import traceback
from typing import Optional

# ─── Import core logic ────────────────────────────────────────────
from chains.general_chain import get_llm, get_conversation_chain
from chains.academic_rag import (
    build_vector_store_for_pdf,
    get_academic_chain_for_pdfs,
    is_query_relevant_to_pdfs,
    # legacy (kept for backward compat — not used in new flow)
    build_vector_store,
    get_academic_chain,
    is_query_relevant_to_docs,
    academic_query,
)
from memory.conversation_memory import get_memory, search_past_conversations
from notes.notes_store import save_note, get_all_notes
from chains.router import get_router_chain, classify_input

# ─── FastAPI App ───────────────────────────────────────────────────
app = FastAPI(
    title="Personal AI Chatbot",
    description="A smart personal assistant with academic RAG, notes, and general chat.",
    version="2.0.0"
)

# ─── Initialize core components ───────────────────────────────────
llm = get_llm()
router_chain = get_router_chain()

academic_docs_dir = "data/academic_docs"
os.makedirs(academic_docs_dir, exist_ok=True)

# ─── Per-session state ────────────────────────────────────────────
# session_memories:  session_id → ConversationBufferMemory
# session_pdfs:      session_id → list[str] of PDF filenames UPLOADED in that session
session_memories: dict = {}
session_pdfs: dict = {}    # only PDFs uploaded THIS session are active

DEFAULT_SESSION = "default"


def get_session_memory(session_id: str):
    """Return (or lazily create) the per-session memory object."""
    if session_id not in session_memories:
        session_memories[session_id] = get_memory(session_id)
    return session_memories[session_id]


def get_session_chain(session_id: str):
    """Return a conversation chain backed by this session's memory."""
    mem = get_session_memory(session_id)
    return get_conversation_chain(llm, mem)


def session_active_pdfs(session_id: str) -> list:
    """Return list of PDFs uploaded/active in this session."""
    return session_pdfs.get(session_id, [])


# ─── Pydantic Models ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = DEFAULT_SESSION


class ChatResponse(BaseModel):
    response: str
    category: str


class NoteRequest(BaseModel):
    content: str


class NewChatRequest(BaseModel):
    session_id: Optional[str] = DEFAULT_SESSION


# ─── Helpers ──────────────────────────────────────────────────────
def safe_str(value):
    """Safely convert chain response to string."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            part.get("text", str(part)) if isinstance(part, dict) else str(part)
            for part in value
        )
    if hasattr(value, 'content'):
        return safe_str(value.content)
    return str(value)


def is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception is a Gemini/Google 429 quota error."""
    msg = str(exc).lower()
    return (
        "resource_exhausted" in msg
        or "quota" in msg
        or "429" in msg
        or "rate limit" in msg
        or "rateerror" in msg
    )


RATE_LIMIT_MESSAGE = (
    "⚠️ I've hit my API rate limit for now — the free tier has a small quota. "
    "Please wait about 30–60 seconds and try again. "
    "If this keeps happening, consider upgrading your Gemini API plan."
)


# ─── API Endpoints ────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "Personal AI Assistant v2"
    }


@app.post("/api/new-chat")
async def new_chat(request: NewChatRequest):
    """
    Reset server-side conversation memory for a session.
    History FILE is kept (for cross-session memory search).
    PDFs are cleared so the new chat starts without any active PDF context.
    """
    sid = request.session_id or DEFAULT_SESSION
    # Remove in-memory chain so new session gets a fresh memory object
    if sid in session_memories:
        del session_memories[sid]
    # Clear active PDFs for this session (user must re-upload to activate)
    if sid in session_pdfs:
        del session_pdfs[sid]
    return {"status": "ok", "message": "New chat session started."}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    Routing logic:
      A. Session HAS active PDFs:
         1. Check if query is relevant to those PDFs → answer from THOSE PDFs only
         2. Not relevant → general conversation
      B. Session has NO active PDFs:
         1. NOTES category → notes store
         2. Otherwise → general conversation
            + if query matches past conversations → inject that context first
    """
    sid = request.session_id or DEFAULT_SESSION
    user_input = request.message.strip()

    if not user_input:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session_chain = get_session_chain(sid)
    active_pdfs = session_active_pdfs(sid)

    try:
        # ── Classify intent ──────────────────────────────────────
        try:
            category = classify_input(router_chain, user_input)
        except AttributeError:
            raw_response = router_chain.invoke({"user_input": user_input})
            content = raw_response.content
            if isinstance(content, list):
                content = "".join(
                    part.get("text", str(part)) if isinstance(part, dict) else str(part)
                    for part in content
                )
            category = content.strip().upper()
            if category not in ["ACADEMIC", "NOTES", "GENERAL"]:
                category = "GENERAL"

        # ══════════════════════════════════════════════════════════
        # PATH A: Session has active PDFs → answer only from them
        # ══════════════════════════════════════════════════════════
        if active_pdfs:
            # Always check relevance first — don't force PDF answer for
            # unrelated questions (e.g. "how are you")
            if is_query_relevant_to_pdfs(user_input, active_pdfs):
                try:
                    pdf_chain = get_academic_chain_for_pdfs(llm, active_pdfs)
                    response = safe_str(academic_query(pdf_chain, user_input))
                    category = "ACADEMIC"
                except Exception as pdf_err:
                    if is_rate_limit_error(pdf_err):
                        raise
                    # Fallback to general if PDF chain fails
                    response = safe_str(session_chain.predict(input=user_input))
                    category = "GENERAL"
            else:
                # Query not about the PDF → general answer
                response = safe_str(session_chain.predict(input=user_input))
                category = "GENERAL"

        # ══════════════════════════════════════════════════════════
        # PATH B: No active PDFs — notes or general + past memory
        # ══════════════════════════════════════════════════════════
        else:
            if category == "NOTES":
                if user_input.lower().startswith("remember:"):
                    note_content = user_input[len("remember:"):].strip()
                    if note_content:
                        save_note(note_content)
                        response = "Got it! I've saved that to your personal notes. 📝"
                    else:
                        response = "Please provide some content after 'remember:' to save."
                else:
                    notes_context = get_all_notes()
                    prompt_with_notes = (
                        f"Here are my personal notes:\n{notes_context}\n\n"
                        f"Answer the user based on these notes. User: {user_input}"
                    )
                    response = safe_str(session_chain.predict(input=prompt_with_notes))
            else:
                # General chat — also search past conversations (GPT-style memory)
                past_context = search_past_conversations(user_input, sid)
                if past_context:
                    augmented_input = (
                        f"{past_context}\n\n"
                        f"Now answer the user's current question using the above context "
                        f"if relevant:\n{user_input}"
                    )
                    response = safe_str(session_chain.predict(input=augmented_input))
                else:
                    response = safe_str(session_chain.predict(input=user_input))
                category = "GENERAL"

        return ChatResponse(response=response, category=category)

    except Exception as e:
        traceback.print_exc()
        if is_rate_limit_error(e):
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
        raise HTTPException(
            status_code=500,
            detail="Something went wrong on my end. Please try again."
        )


@app.post("/api/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Form(DEFAULT_SESSION)
):
    """
    Upload a PDF file and build its own isolated FAISS index.
    Tags the PDF to the current session so the session becomes PDF-aware.
    Other sessions are NOT affected.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save the PDF to disk (replaces existing file with same name)
        file_path = os.path.join(academic_docs_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Build an isolated FAISS index for this PDF only
        build_vector_store_for_pdf(file_path)

        # Register this PDF under the current session (REPLACES any previous active PDF)
        session_pdfs[session_id] = [file.filename]

        return {
            "status": "success",
            "message": f"'{file.filename}' uploaded and indexed successfully!",
            "filename": file.filename,
            "active_pdfs": session_pdfs[session_id]
        }
    except Exception as e:
        traceback.print_exc()
        if is_rate_limit_error(e):
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.delete("/api/delete-pdf")
async def delete_pdf(filename: str, session_id: str = DEFAULT_SESSION):
    """
    Delete a specific PDF and its FAISS index.
    Removes it from all sessions that had it active.
    """
    import shutil as _shutil
    import re

    if not filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files can be deleted.")

    file_path = os.path.join(academic_docs_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"'{filename}' not found.")

    try:
        # Remove the PDF file
        os.remove(file_path)

        # Remove its FAISS index directory
        stem = re.sub(r'[^\w\-]', '_', os.path.splitext(filename)[0])
        store_path = os.path.join("data/vector_store", stem)
        if os.path.exists(store_path):
            _shutil.rmtree(store_path)

        # Remove from all session tracking
        for sid in list(session_pdfs.keys()):
            if filename in session_pdfs[sid]:
                session_pdfs[sid].remove(filename)

        remaining = [f for f in os.listdir(academic_docs_dir) if f.endswith('.pdf')]
        return {
            "status": "success",
            "message": f"'{filename}' deleted successfully!",
            "remaining": remaining
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error deleting PDF: {str(e)}")


@app.get("/api/notes")
async def get_notes():
    """Get all saved notes."""
    try:
        notes_text = get_all_notes()
        return {"notes": notes_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/notes")
async def create_note(request: NoteRequest):
    """Save a new note."""
    content = request.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Note content cannot be empty")
    try:
        save_note(content)
        return {"status": "success", "message": "Note saved successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/uploaded-pdfs")
async def get_uploaded_pdfs():
    """List all uploaded PDFs."""
    try:
        pdfs = [f for f in os.listdir(academic_docs_dir) if f.endswith('.pdf')]
        return {"pdfs": pdfs, "count": len(pdfs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Serve Static Frontend ────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


