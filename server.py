from pathlib import Path
import os

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from pydantic import BaseModel

import rag_engine
import session_store
from config import CHROMA_PATH, COLLECTION_NAME
from ingest import process_pdf

try:
    import chromadb
    _db_client = chromadb.PersistentClient(path=CHROMA_PATH)
    _knowledge_collection = _db_client.get_collection(COLLECTION_NAME)
except Exception:
    _knowledge_collection = None

app = FastAPI(title="Nallas AI")


class ChatRequest(BaseModel):
    session_id: str
    question: str


def _storage_info():
    backend_status = session_store.get_backend_status()
    return {
        "knowledge_base_chunks": _knowledge_collection.count() if _knowledge_collection else 0,
        "long_term_memory": rag_engine.get_memory_count(),
        "session_store": session_store.get_session_count(),
        "current_session_turns": 0,
        "embedding_model": "all-MiniLM-L6-v2",
        "uploaded_documents": len(list(Path("data").glob("*.pdf"))) if Path("data").exists() else 0,
        "session_backend": backend_status["backend"],
        "session_backend_reason": backend_status["reason"],
    }


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Nallas AI</title>
  <style>
    :root {
      --primary: #2A3A86;
      --secondary: #4C63D2;
      --accent: #756CA1;
      --bg: #111827;
      --sidebar: #1A1F2E;
      --card: #242B3D;
      --text: #F8FAFC;
      --border: #394867;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background: radial-gradient(circle at top, rgba(76,99,210,.16), transparent 34%), var(--bg); color: var(--text); }
    .layout { display: grid; grid-template-columns: 300px 1fr; min-height: 100vh; }
    .sidebar { background: linear-gradient(180deg, #1A1F2E, #151B28); border-right: 1px solid var(--border); padding: 20px; }
    .brand { letter-spacing: .24em; text-transform: uppercase; font-size: .78rem; color: rgba(248,250,252,.7); margin-bottom: 10px; }
    h1 { margin: 0 0 12px; font-size: 30px; }
    .btn { width: 100%; background: linear-gradient(135deg, var(--primary), var(--secondary)); color: white; border: 0; border-radius: 14px; padding: 12px 14px; cursor: pointer; font-weight: 600; }
    .btn.secondary { background: rgba(36,43,61,.8); border: 1px solid var(--border); }
    .session-list { display: grid; gap: 10px; margin-top: 14px; }
    .session { background: rgba(36,43,61,.7); border: 1px solid rgba(57,72,103,.8); border-radius: 16px; padding: 12px; cursor: pointer; }
    .session.active { border-color: var(--secondary); background: rgba(42,58,134,.28); }
    .session .title { font-weight: 700; margin-bottom: 4px; }
    .session .meta { font-size: 12px; opacity: .75; }
    .card-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 18px; }
    .card { background: rgba(36,43,61,.88); border: 1px solid var(--border); border-radius: 16px; padding: 12px; }
    .card .label { font-size: 12px; opacity: .72; }
    .card .value { font-size: 18px; font-weight: 700; margin-top: 6px; }
    .main { display: grid; grid-template-rows: auto 1fr auto; }
    .topbar { padding: 18px 22px; border-bottom: 1px solid rgba(57,72,103,.55); }
    .chat { padding: 22px; overflow-y: auto; display: grid; gap: 14px; }
    .bubble { max-width: 75%; border: 1px solid rgba(57,72,103,.82); border-radius: 20px; padding: 14px 16px; box-shadow: 0 12px 28px rgba(0,0,0,.18); white-space: pre-wrap; }
    .user { justify-self: end; background: linear-gradient(135deg, rgba(42,58,134,.94), rgba(76,99,210,.94)); }
    .assistant { justify-self: start; background: linear-gradient(180deg, rgba(36,43,61,.96), rgba(29,35,49,.96)); }
    .composer { padding: 16px 22px 24px; border-top: 1px solid rgba(57,72,103,.55); display: grid; gap: 10px; }
    textarea, input[type=text] { width: 100%; background: #242B3D; color: var(--text); border: 1px solid var(--border); border-radius: 16px; padding: 12px 14px; }
    .row { display: flex; gap: 10px; }
    .row .btn { width: auto; flex: 0 0 auto; }
    .source { display: inline-block; margin-top: 8px; padding: 6px 10px; border-radius: 999px; border: 1px solid rgba(117,108,161,.55); background: rgba(17,24,39,.46); font-size: 12px; }
    .muted { opacity: .7; font-size: 13px; }
    .upload { margin-top: 18px; display: grid; gap: 10px; }
    .file { background: rgba(36,43,61,.55); border: 1px solid var(--border); padding: 10px; border-radius: 14px; }
    @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } .sidebar { order: 2; } .bubble { max-width: 100%; } }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="brand">Nallas Technologies</div>
      <h1>NALLAS AI</h1>
      <button class="btn" id="newChat">+ New Chat</button>
      <div class="session-list" id="sessionList"></div>
      <div class="upload">
        <div class="file">
          <input id="uploadInput" type="file" accept=".pdf">
          <button class="btn secondary" id="uploadBtn" style="margin-top:10px;">Upload Document</button>
          <div class="muted" id="uploadStatus"></div>
        </div>
      </div>
      <div class="card-grid" id="storageInfo"></div>
    </aside>
    <main class="main">
      <div class="topbar">
        <div style="font-size:14px; opacity:.72;">Session</div>
        <div id="sessionTitle" style="font-size:24px; font-weight:800;">Untitled Session</div>
        <div class="muted" id="sessionMeta"></div>
      </div>
      <div class="chat" id="chat"></div>
      <div class="composer">
        <textarea id="question" rows="3" placeholder="Ask a question..."></textarea>
        <div class="row">
          <button class="btn" id="sendBtn">Send</button>
          <button class="btn secondary" id="refreshBtn">Refresh</button>
        </div>
      </div>
    </main>
  </div>
  <script>
    const state = { sessionId: localStorage.getItem("nallas_session_id") || null, sessions: [] };
    const el = id => document.getElementById(id);

    function setSession(sessionId) {
      state.sessionId = sessionId;
      localStorage.setItem("nallas_session_id", sessionId);
    }

    async function api(path, opts={}) {
      const res = await fetch(path, opts);
      return await res.json();
    }

    function fmt(ts) {
      if (!ts) return "Just now";
      return new Date(ts * 1000).toLocaleString();
    }

    function renderChat(turns) {
      const chat = el("chat");
      chat.innerHTML = "";
      turns.forEach(turn => {
        const user = document.createElement("div");
        user.className = "bubble user";
        user.textContent = turn.user_msg;
        chat.appendChild(user);
        const assistant = document.createElement("div");
        assistant.className = "bubble assistant";
        assistant.textContent = turn.assistant_display_msg || turn.assistant_msg || "";
        chat.appendChild(assistant);
        (turn.assistant_sources || []).forEach(src => {
          const chip = document.createElement("div");
          chip.className = "source";
          chip.textContent = "PDF " + src;
          assistant.appendChild(chip);
        });
      });
      chat.scrollTop = chat.scrollHeight;
    }

    async function loadSessions() {
      const data = await api("/api/sessions");
      state.sessions = data.sessions;
      const list = el("sessionList");
      list.innerHTML = "";
      data.sessions.forEach(sess => {
        const item = document.createElement("div");
        item.className = "session" + (sess.session_id === state.sessionId ? " active" : "");
        item.innerHTML = `<div class="title">${sess.title}</div><div class="meta">${sess.turn_count} turns • ${fmt(sess.last_timestamp)}</div>`;
        item.onclick = async () => {
          setSession(sess.session_id);
          await loadSession(sess.session_id);
        };
        list.appendChild(item);
      });
      renderStorage(data.storage);
      if (!state.sessionId && data.sessions[0]) {
        setSession(data.sessions[0].session_id);
      }
      if (state.sessionId) await loadSession(state.sessionId);
    }

    function renderStorage(storage) {
      const grid = el("storageInfo");
      grid.innerHTML = `
        <div class="card"><div class="label">Knowledge Base</div><div class="value">${storage.knowledge_base_chunks}</div></div>
        <div class="card"><div class="label">Long-Term Memory</div><div class="value">${storage.long_term_memory}</div></div>
        <div class="card"><div class="label">Session Store</div><div class="value">${storage.session_store}</div></div>
        <div class="card"><div class="label">Current Session</div><div class="value">${storage.current_session_turns}</div></div>
        <div class="card"><div class="label">Embedding Model</div><div class="value">all-MiniLM-L6-v2</div></div>
        <div class="card"><div class="label">Uploaded Docs</div><div class="value">${storage.uploaded_documents}</div></div>
        <div class="card" style="grid-column:1 / -1;"><div class="label">Session Backend</div><div class="value">${storage.session_backend}</div><div class="muted">${storage.session_backend_reason || ""}</div></div>
      `;
    }

    async function loadSession(sessionId) {
      const data = await api(`/api/sessions/${sessionId}`);
      el("sessionTitle").textContent = data.title;
      el("sessionMeta").textContent = `${data.turn_count} turns • Last updated ${fmt(data.last_timestamp)}`;
      renderChat(data.turns);
      document.querySelectorAll(".session").forEach((node, idx) => {
        node.classList.toggle("active", state.sessions[idx] && state.sessions[idx].session_id === sessionId);
      });
      const res = await api("/api/storage");
      renderStorage(res);
    }

    el("newChat").onclick = async () => {
      const res = await api("/api/sessions/new", { method: "POST" });
      setSession(res.session_id);
      await loadSessions();
    };

    el("sendBtn").onclick = async () => {
      const question = el("question").value.trim();
      if (!question) return;
      if (!state.sessionId) {
        const res = await api("/api/sessions/new", { method: "POST" });
        setSession(res.session_id);
      }
      const data = await api("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId, question })
      });
      el("question").value = "";
      await loadSessions();
      await loadSession(data.session_id);
    };

    el("refreshBtn").onclick = loadSessions;

    el("uploadBtn").onclick = async () => {
      const file = el("uploadInput").files[0];
      if (!file) return;
      const form = new FormData();
      form.append("file", file);
      el("uploadStatus").textContent = "Uploading...";
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const data = await res.json();
      el("uploadStatus").textContent = data.message;
      await loadSessions();
    };

    loadSessions();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.get("/api/sessions")
def sessions():
    recent = session_store.list_recent_sessions(limit=5)
    active_session_id = recent[0]["session_id"] if recent else None
    storage = _storage_info()
    if active_session_id:
        storage["current_session_turns"] = session_store.get_session_turn_count(active_session_id)
    return JSONResponse({"sessions": recent, "storage": storage})


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str):
    summary = session_store.get_session_summary(session_id)
    turns = session_store.get_session_turns(session_id)
    return JSONResponse({**summary, "turns": turns})


@app.post("/api/sessions/new")
def new_session():
    session_id = session_store.create_session()
    return JSONResponse({"session_id": session_id})


@app.post("/api/sessions/{session_id}/rename")
def rename(session_id: str, title: str = Form(...)):
    return JSONResponse({"ok": session_store.rename_session(session_id, title)})


@app.post("/api/chat")
def chat(payload: ChatRequest):
    answer, sources = rag_engine.ask_question(payload.question, session_id=payload.session_id)
    return JSONResponse({
        "session_id": payload.session_id,
        "answer": answer,
        "sources": sources,
        "summary": session_store.get_session_summary(payload.session_id),
    })


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)
    save_path = Path("data") / file.filename
    content = await file.read()
    save_path.write_bytes(content)
    chunk_count = process_pdf(str(save_path))
    return JSONResponse({"message": f"Indexed successfully ({chunk_count} chunks)"})


@app.get("/api/storage")
def storage():
    return JSONResponse(_storage_info())


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
