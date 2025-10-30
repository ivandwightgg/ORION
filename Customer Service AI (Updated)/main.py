from __future__ import annotations
import asyncio
import os
import sys
import yaml
from pathlib import Path
from typing import List, Optional, Iterable, Tuple, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiofiles

# --- FILESYSTEM SAFETY GUARDS ---
def _is_bad_path(p: str) -> bool:
    """Reject unsafe paths like systemd-private or Proton Z: mappings."""
    return (
        "systemd-private-" in p
        or ("/steamapps/compatdata/" in p and "/pfx/dosdevices/z:" in p)
    )


def _safe_walk(top: str, topdown: bool = True, onerror=None, followlinks: bool = False) -> Iterable[Tuple[str, list, list]]:
    """Safe os.walk variant that avoids symlinks, Proton, and permission errors."""
    followlinks = False
    try:
        top = os.fspath(top)
    except Exception:
        top = str(top)

    if _is_bad_path(top):
        return

    try:
        with os.scandir(top) as it:
            dirs, nondirs = [], []
            for entry in it:
                path = os.path.join(top, entry.name)
                if _is_bad_path(path):
                    continue
                try:
                    if entry.is_symlink():
                        if entry.is_file(follow_symlinks=False):
                            nondirs.append(entry.name)
                    elif entry.is_dir(follow_symlinks=False):
                        dirs.append(entry.name)
                    else:
                        nondirs.append(entry.name)
                except (PermissionError, FileNotFoundError):
                    continue
    except (NotADirectoryError, FileNotFoundError, PermissionError) as e:
        if onerror:
            try:
                onerror(e)
            except Exception:
                pass
        return

    if topdown:
        yield top, dirs, nondirs

    for d in list(dirs):
        new_path = os.path.join(top, d)
        if _is_bad_path(new_path):
            continue
        for x in _safe_walk(new_path, topdown, onerror, False):
            yield x

    if not topdown:
        yield top, dirs, nondirs


# Monkey patch globally (still necessary for other modules)
_os_walk_original = os.walk
os.walk = _safe_walk  # type: ignore

def _safe_rglob(self: Path, pattern: str):
    for root_dir, _, files in _safe_walk(str(self)):
        for fname in files:
            try:
                p = Path(root_dir) / fname
                if p.match(pattern):
                    yield p
            except Exception:
                continue

_Path_rglob_original = Path.rglob
Path.rglob = _safe_rglob  # type: ignore
# --- END FS SAFETY GUARDS ---


# --- CONFIGURATION ---
def load_config(path: str = "config.yaml") -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Invalid YAML structure.")
        return cfg
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing config file: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parse error: {e}")


CONFIG = load_config()
ING = CONFIG.get("ingest", {})

app = FastAPI(title="Customer Service AI — Local (Ollama)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG.get("cors", {}).get("origins", ["http://localhost:8000"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- IMPORT LOCAL MODULES ---
from .orchestrator import Orchestrator
from .rag import RAG
from .web_utils import fetch_page_text
from .ingest_watcher import run_watcher, scan_all


# --- APP INITIALIZATION ---
rag = RAG()
orch = Orchestrator(rag=rag)
bg_task: Optional[asyncio.Task] = None


@app.on_event("startup")
async def on_startup():
    print("[fs-guard] Active: safe os.walk + Path.rglob overrides", file=sys.stderr)
    for key in ["uploads_dir", "links_dir", "messages_dir", "links_cache_dir", "root"]:
        os.makedirs(ING.get(key, "data"), exist_ok=True)

    global bg_task
    bg_task = asyncio.create_task(
        run_watcher(rag, ING.get("scan_interval_seconds", 10))
    )


@app.on_event("shutdown")
async def on_shutdown():
    global bg_task
    if bg_task and not bg_task.done():
        bg_task.cancel()
        try:
            await asyncio.shield(bg_task)
        except asyncio.CancelledError:
            pass


# --- Pydantic Schemas ---
class ChatIn(BaseModel):
    user_id: str
    message: str
    namespace: Optional[str] = None


class TeachIn(BaseModel):
    text: str
    namespace: Optional[str] = None


class LinksIn(BaseModel):
    urls: List[str]
    namespace: Optional[str] = None


# --- Routes ---
@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/ui", status_code=307)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return HTMLResponse(status_code=204)


@app.post("/teach")
async def teach(payload: TeachIn):
    """Save text snippets to a namespace for model ingestion."""
    ns = payload.namespace or ING.get("namespace_default", "default")
    ns_dir = Path(ING["messages_dir"]) / ns
    ns_dir.mkdir(parents=True, exist_ok=True)
    path = ns_dir / "note-ui.txt"

    try:
        async with aiofiles.open(path, "a", encoding="utf-8") as f:
            await f.write(f"\n{payload.text.strip()}\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save note: {e}")

    return {"status": "ok", "saved": str(path), "namespace": ns}


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), namespace: Optional[str] = Form(None)):
    """Handle user file uploads."""
    ns = namespace or ING.get("namespace_default", "default")
    ns_dir = Path(ING["uploads_dir"]) / ns
    ns_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        dest = ns_dir / f.filename
        try:
            async with aiofiles.open(dest, "wb") as out:
                await out.write(await f.read())
            saved.append(str(dest))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save {f.filename}: {e}")

    return {"status": "ok", "saved": saved, "namespace": ns}


@app.post("/links")
async def links(payload: LinksIn):
    """Save URL links for ingestion."""
    ns = payload.namespace or ING.get("namespace_default", "default")
    ns_dir = Path(ING["links_dir"]) / ns
    ns_dir.mkdir(parents=True, exist_ok=True)
    path = ns_dir / "links-ui.txt"

    try:
        async with aiofiles.open(path, "a", encoding="utf-8") as f:
            for u in payload.urls:
                await f.write(u.strip() + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write links: {e}")

    return {"status": "ok", "saved": str(path), "count": len(payload.urls), "namespace": ns}


@app.post("/ingest/scan")
async def ingest_scan():
    """Manually trigger ingestion scan."""
    totals = await scan_all(rag)
    return {"status": "ok", "totals": totals}


@app.post("/chat")
async def chat(payload: ChatIn):
    """Handle user messages and return LLM responses via the orchestrator."""
    try:
        result = await orch.handle_chat(
            user_id=payload.user_id,
            message=payload.message,
            namespace=payload.namespace,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat handling failed: {e}")


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """Minimal in-browser chat interface."""
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Chat</title>
<style>
  :root { --bg:#0b0c10; --panel:#14161b; --text:#e6e6e6; --muted:#9aa0a6; --accent:#6aa0ff; }
  *{ box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:16px/1.45 system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif; }
  header { padding:14px 16px; background:var(--panel); border-bottom:1px solid #22262d; position:sticky; top:0; z-index:1; }
  header h1 { margin:0; font-size:16px; font-weight:600; }
  #chat { max-width:900px; margin:0 auto; padding:16px; min-height:calc(100vh - 140px); }
  .bubble { max-width:78%; padding:10px 12px; border-radius:14px; margin:8px 0; white-space:pre-wrap; word-break:break-word; }
  .u { background:#1f232b; margin-left:auto; border-bottom-right-radius:6px; }
  .a { background:#11151b; border:1px solid #22262d; border-bottom-left-radius:6px; }
  .sys { color: var(--muted); text-align:center; margin:12px 0; font-size:13px; }
  footer { position:sticky; bottom:0; background:var(--panel); border-top:1px solid #22262d; }
  .row { display:flex; gap:8px; padding:12px; max-width:900px; margin:0 auto; }
  #msg { flex:1; padding:10px 12px; border-radius:12px; background:#0e1015; color:var(--text); border:1px solid #22262d; }
  button { padding:10px 14px; border-radius:12px; border:1px solid #2a313b; background:#1b212a; color:#fff; cursor:pointer; }
  button:hover { background:#212833; }
</style>
</head>
<body>
  <header><h1>Customer Support Assistant</h1></header>
  <div id="chat"></div>
  <footer>
    <div class="row">
      <input id="msg" type="text" placeholder="Type a message and press Enter..." autofocus />
      <button onclick="sendMsg()">Send</button>
    </div>
  </footer>

<script>
const chat = document.getElementById('chat');
const msg = document.getElementById('msg');

function addBubble(text, who){
  const div = document.createElement('div');
  div.className = 'bubble ' + (who === 'u' ? 'u' : 'a');
  div.textContent = text;
  chat.appendChild(div);
  window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
}

function addSystem(text){
  const d = document.createElement('div');
  d.className = 'sys';
  d.textContent = text;
  chat.appendChild(d);
  window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
}

async function sendMsg(){
  const text = msg.value.trim();
  if(!text) return;
  addBubble(text, 'u');
  msg.value='';
  addSystem('…thinking…');

  try {
    const r = await fetch('/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({user_id:'ui', message: text, namespace: null})
    });
    const raw = await r.text();
    const last = chat.querySelector('.sys:last-child');
    if(last && last.textContent.includes('…thinking…')) last.remove();

    try {
      const json = JSON.parse(raw);
      addBubble(json?.answer ?? '(no answer)', 'a');
    } catch(e){
      addBubble(`Non-JSON response (status ${r.status}):\\n\\n` + raw, 'a');
    }
  } catch (e) {
    const last = chat.querySelector('.sys:last-child');
    if(last && last.textContent.includes('…thinking…')) last.remove();
    addBubble('Request failed: ' + (e?.message || e), 'a');
  }
}

msg.addEventListener('keydown', (ev)=>{
  if(ev.key === 'Enter' && !ev.shiftKey){
    ev.preventDefault();
    sendMsg();
  }
});
</script>
</body>
</html>
"""
