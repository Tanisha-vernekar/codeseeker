"""A single-page web interface for codeseeker.

Run it with ``codeseeker web`` (or ``python -m codeseeker web``) and it starts a
local server and opens your browser. From the page you can index a local folder
or a remote ``owner/repo``, then search, explain, ask questions, and view stats
— all the CLI features, in the browser.

Flask is an optional dependency; install it with ``pip install codeseeker[web]``.
"""

from __future__ import annotations

import os
import threading
from collections import Counter

from codeseeker.index import CodeIndex, default_index_dir
from codeseeker.qa import answer_question
from codeseeker.repo import DEFAULT_CACHE_DIR, resolve_source
from codeseeker.summary import summarize_repo


class _State:
    """Server-side singleton holding the currently loaded index."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.index: CodeIndex | None = None
        self.origin: str = ""
        self.local_path: str = ""

    def set_index(self, index: CodeIndex, origin: str, local_path: str) -> None:
        with self.lock:
            self.index = index
            self.origin = origin
            self.local_path = local_path


STATE = _State()


def _index_stats(index: CodeIndex) -> dict:
    from codeseeker.llm import LLMClient

    languages = Counter(c.language for c in index.chunks)
    kinds = Counter(c.kind for c in index.chunks)
    files = {c.path for c in index.chunks}
    dim = int(index.vectors.shape[1]) if index.vectors.ndim == 2 and index.vectors.size else 0
    return {
        "origin": index.origin,
        "root": index.root,
        "is_remote": index.is_remote,
        "num_files": len(files),
        "num_chunks": len(index),
        "embedder": index.embedder.name,
        "engine_name": _friendly_engine_name(index.embedder.name),
        "search_backend": index.backend_name,
        "vector_dim": dim,
        "llm_available": LLMClient().available(),
        "languages": languages.most_common(),
        "kinds": dict(kinds),
    }


def _code_map(index: CodeIndex, max_files: int = 25, max_symbols_per_file: int = 8) -> list[dict]:
    """Return a lightweight architecture map grouped by file."""
    grouped: dict[str, list] = {}
    for chunk in index.chunks:
        if not chunk.symbol:
            continue
        grouped.setdefault(chunk.path, []).append(chunk)

    rows = []
    for path, chunks in grouped.items():
        # keep unique symbol names in file order
        seen = set()
        symbols = []
        for c in sorted(chunks, key=lambda x: (x.start_line, x.end_line)):
            if c.symbol in seen:
                continue
            seen.add(c.symbol)
            symbols.append({"name": c.symbol, "kind": c.kind, "line": c.start_line})
        rows.append({"path": path, "symbols": symbols[:max_symbols_per_file], "total_symbols": len(symbols)})

    rows.sort(key=lambda r: (-r["total_symbols"], r["path"]))
    return rows[:max_files]


def _friendly_engine_name(name: str) -> str:
    if name == "tfidf":
        return "Fast & simple (offline)"
    if name == "sentence-transformers":
        return "Deep semantic (neural)"
    return name


def create_app():
    """Create and configure the Flask application."""
    try:
        from flask import Flask, jsonify, request
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Flask is required for the web UI. Install it with: pip install codeseeker[web]"
        ) from exc

    app = Flask(__name__)

    @app.get("/")
    def home():
        return INDEX_HTML

    @app.get("/api/status")
    def status():
        with STATE.lock:
            if STATE.index is None:
                return jsonify({"loaded": False})
            return jsonify({"loaded": True, "stats": _index_stats(STATE.index)})

    @app.post("/api/index")
    def api_index():
        data = request.get_json(silent=True) or {}
        source = (data.get("source") or "").strip()
        if not source:
            return jsonify({"error": "Please provide a repository path or URL."}), 400

        backend = data.get("backend") or "auto"
        faiss_pref = {"auto": "auto", "on": True, "off": False}.get(data.get("faiss", "auto"), "auto")
        exts = data.get("ext")
        extensions = [e.strip() for e in exts.split(",") if e.strip()] if exts else None
        clone_local = bool(data.get("clone_local", True))
        clone_dir = (data.get("clone_dir") or "").strip()
        cache_dir = clone_dir or (os.path.abspath("downloaded_repos") if clone_local else DEFAULT_CACHE_DIR)

        try:
            repo = resolve_source(
                source,
                cache_dir=cache_dir,
                update=bool(data.get("update")),
            )
        except (FileNotFoundError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            index = CodeIndex.build(
                root=repo.local_path,
                extensions=extensions,
                backend=backend,
                origin=repo.origin,
                is_remote=repo.is_remote,
                prefer_faiss=faiss_pref,
            )
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"Indexing failed: {exc}"}), 500

        # Best-effort persistence so it survives restarts.
        try:
            index.save(default_index_dir(repo.local_path))
        except Exception:  # noqa: BLE001
            pass

        STATE.set_index(index, repo.origin, repo.local_path)
        return jsonify(
            {
                "ok": True,
                "cloned": repo.cloned,
                "is_remote": repo.is_remote,
                "cache_dir": cache_dir if repo.is_remote else "",
                "local_path": repo.local_path,
                "stats": _index_stats(index),
                "engine_name": _friendly_engine_name(index.embedder.name),
            }
        )

    @app.post("/api/search")
    def api_search():
        with STATE.lock:
            index = STATE.index
        if index is None:
            return jsonify({"error": "No index loaded. Index a repository first."}), 400
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"error": "Please enter a search query."}), 400

        def _list(value):
            return [v.strip() for v in value.split(",") if v.strip()] if value else None

        results = index.search(
            query,
            top_k=int(data.get("top_k", 8)),
            languages=_list(data.get("lang")),
            kinds=_list(data.get("kind")),
            path_contains=(data.get("path") or None),
            mode=(data.get("mode") or "hybrid"),
            semantic_weight=float(data.get("semantic_weight", 0.8)),
        )
        return jsonify({"results": [r.to_dict() for r in results]})

    @app.post("/api/explain")
    def api_explain():
        with STATE.lock:
            index = STATE.index
        if index is None:
            return jsonify({"error": "No index loaded. Index a repository first."}), 400
        summary = summarize_repo(index, root=index.root)
        return jsonify(summary.to_dict())

    @app.post("/api/ask")
    def api_ask():
        with STATE.lock:
            index = STATE.index
        if index is None:
            return jsonify({"error": "No index loaded. Index a repository first."}), 400
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Please enter a question."}), 400
        result = answer_question(index, question, top_k=int(data.get("top_k", 6)))
        return jsonify(result.to_dict())

    @app.get("/api/stats")
    def api_stats():
        with STATE.lock:
            index = STATE.index
        if index is None:
            return jsonify({"error": "No index loaded."}), 400
        return jsonify(_index_stats(index))

    @app.get("/api/suggestions")
    def api_suggestions():
        with STATE.lock:
            index = STATE.index
        if index is None:
            return jsonify({"error": "No index loaded."}), 400
        from codeseeker.analysis import analyze_repo
        from codeseeker.summary import _clean_project_name, suggest_ask_questions, suggest_search_queries

        root = index.root or "."
        name = _clean_project_name(getattr(index, "origin", "") or "", root)
        profile = analyze_repo(index, root, name)
        return jsonify({
            "questions": suggest_ask_questions(index, profile),
            "ask_questions": suggest_ask_questions(index, profile),
            "search_queries": suggest_search_queries(index, profile),
        })

    @app.get("/api/map")
    def api_map():
        with STATE.lock:
            index = STATE.index
        if index is None:
            return jsonify({"error": "No index loaded."}), 400
        return jsonify({"files": _code_map(index)})

    return app


def run_server(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    """Start the web server (and optionally open the default browser)."""
    app = create_app()
    url = f"http://{host}:{port}"
    if open_browser:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"codeseeker web UI running at {url}  (press Ctrl-C to stop)")
    # threaded=True so the UI stays responsive (e.g. status polls) while a
    # large repository is being cloned/indexed in another request.
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


# --------------------------------------------------------------------------- #
# The single-page UI (plain string; no f-string so braces stay literal).
# --------------------------------------------------------------------------- #
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>codeseeker</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --panel2: #1c2230; --border: #2a3140;
    --text: #e6edf3; --muted: #8b949e; --accent: #6ea8fe; --accent2: #a371f7;
    --green: #3fb950; --yellow: #d29922; --mono: ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  header { padding: 18px 24px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 14px;
    background: linear-gradient(90deg, #0d1117, #131a2a); position: sticky; top: 0; z-index: 5; }
  header h1 { font-size: 20px; margin: 0; letter-spacing: .3px; }
  header h1 span { background: linear-gradient(90deg,var(--accent),var(--accent2));
    -webkit-background-clip: text; background-clip: text; color: transparent; }
  header .tag { color: var(--muted); font-size: 13px; }
  #statusPill { margin-left: auto; font-size: 12px; color: var(--muted);
    border: 1px solid var(--border); border-radius: 999px; padding: 5px 12px; }
  #statusPill.on { color: var(--green); border-color: #1f6f34; }
  .wrap { max-width: 1000px; margin: 0 auto; padding: 24px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 18px; margin-bottom: 20px; }
  .card h2 { margin: 0 0 12px; font-size: 15px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }
  .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  input, select, textarea, button { font: inherit; }
  input[type=text], textarea, select { background: var(--panel2); color: var(--text);
    border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; outline: none; }
  input[type=text]:focus, textarea:focus, select:focus { border-color: var(--accent); }
  .grow { flex: 1 1 260px; }
  .small { flex: 0 0 auto; width: 120px; }
  button { background: linear-gradient(90deg,var(--accent),var(--accent2)); color: #06090f;
    border: 0; border-radius: 8px; padding: 10px 16px; font-weight: 650; cursor: pointer; }
  button.secondary { background: var(--panel2); color: var(--text); border: 1px solid var(--border); font-weight: 500; }
  button:disabled { opacity: .5; cursor: not-allowed; }
  .hint { color: var(--muted); font-size: 12px; margin-top: 8px; }
  .tabs { display: flex; gap: 6px; margin-bottom: 16px; flex-wrap: wrap; }
  .tab { padding: 8px 16px; border-radius: 8px; cursor: pointer; color: var(--muted);
    border: 1px solid transparent; }
  .tab.active { color: var(--text); background: var(--panel); border-color: var(--border); }
  .pane { display: none; }
  .pane.active { display: block; }
  .result { border: 1px solid var(--border); border-radius: 10px; margin-bottom: 12px; overflow: hidden; }
  .result .head { display: flex; gap: 10px; align-items: center; padding: 10px 14px;
    background: var(--panel2); font-size: 13px; flex-wrap: wrap; }
  .score { color: var(--yellow); font-family: var(--mono); }
  .loc { color: var(--accent); font-family: var(--mono); }
  .sym { color: var(--green); }
  pre { margin: 0; padding: 12px 14px; overflow-x: auto; font-family: var(--mono);
    font-size: 12.5px; line-height: 1.5; color: #cdd9e5; background: #0b0f16; white-space: pre; }
  .desc { white-space: pre-wrap; line-height: 1.6; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
  .chip { font-size: 12px; background: var(--panel2); border: 1px solid var(--border);
    border-radius: 999px; padding: 4px 12px; color: var(--muted); }
  .err { color: #ff7b72; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--muted);
    border-top-color: transparent; border-radius: 50%; animation: spin .7s linear infinite; vertical-align: -2px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  a { color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1><span>code</span>seeker</h1>
  <span id="statusPill">no index loaded</span>
</header>

<div class="wrap">
  <div class="card">
    <h2>1 · Load a GitHub repo or local folder</h2>
    <div class="row">
      <input class="grow" id="source" type="text" placeholder="owner/repo  ·  https://github.com/owner/repo  ·  local path (e.g. .)"
             onkeydown="if(event.key==='Enter')doIndex()" />
      <button id="indexBtn" onclick="doIndex()">Index</button>
    </div>
    <div class="hint">Remote repos are downloaded to your machine, then indexed for fast search.</div>
    <div id="indexOut" class="hint"></div>
  </div>

  <div class="card">
    <div class="tabs">
      <div class="tab active" data-pane="explain" onclick="switchTab('explain')">Explain</div>
      <div class="tab" data-pane="search" onclick="switchTab('search')">Search</div>
      <div class="tab" data-pane="ask" onclick="switchTab('ask')">Ask</div>
      <div class="tab" data-pane="stats" onclick="switchTab('stats')">Stats</div>
    </div>

    <div class="pane active" id="pane-explain">
      <div class="row"><button onclick="doExplain()">Explain this project</button></div>
      <div class="hint">Deep analysis: README + config files + architecture layers + semantic code retrieval. Not a raw README paste.</div>
      <div id="explainOut"></div>
    </div>

    <div class="pane" id="pane-search">
      <div class="hint">🔍 <b>Search</b> finds the most relevant code snippets (functions/classes) — like a smart "find in code".</div>
      <div class="row">
        <input class="grow" id="query" type="text" placeholder="Describe the code you want, e.g. &quot;retry an http request with backoff&quot;"
               onkeydown="if(event.key==='Enter')doSearch()" />
        <button onclick="doSearch()">Search</button>
      </div>
      <div class="hint" id="searchSuggestLabel" style="display:none">Looking for something specific? Try:</div>
      <div class="chips" id="searchSuggestChips"></div>
      <div id="searchOut"></div>
    </div>

    <div class="pane" id="pane-ask">
      <div class="hint">💬 <b>Ask</b> answers a question in words and cites the code it used — like asking a teammate.</div>
      <div class="row">
        <input class="grow" id="question" type="text" placeholder="Ask a question, e.g. &quot;how is authentication handled?&quot;"
               onkeydown="if(event.key==='Enter')doAsk()" />
        <button onclick="doAsk()">Ask</button>
      </div>
      <div class="hint" id="suggestLabel" style="display:none">Not sure what to ask? Try one of these:</div>
      <div class="chips" id="suggestChips"></div>
      <div id="askOut"></div>
    </div>

    <div class="pane" id="pane-stats">
      <div class="row"><button onclick="doStats()">Refresh stats</button></div>
      <div id="statsOut"></div>
    </div>
  </div>
</div>

<script>
function esc(s){ return (s||"").replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function el(id){ return document.getElementById(id); }

async function api(path, body){
  const opts = { method: body ? 'POST' : 'GET', headers: {'Content-Type':'application/json'} };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
  return data;
}

function setStatus(stats){
  const pill = el('statusPill');
  if (stats){
    const ai = stats.llm_available ? '  ·  AI answers: on' : '';
    pill.textContent = (stats.origin || stats.root) + '  ·  ' + stats.num_chunks + ' chunks  ·  ' +
      (stats.engine_name || stats.embedder) + ' + ' + stats.search_backend + ai;
    pill.classList.add('on');
  } else { pill.textContent = 'no index loaded'; pill.classList.remove('on'); }
}

async function doIndex(){
  const source = el('source').value.trim();
  if(!source){ el('indexOut').innerHTML = '<span class="err">Enter a repository path or URL.</span>'; return; }
  const btn = el('indexBtn'); btn.disabled = true;
  el('indexOut').innerHTML = '<span class="spinner"></span> Fetching &amp; indexing…';
  try {
    const d = await api('/api/index', { source });
    el('indexOut').innerHTML = '✅ Project loaded. Open the <b>Explain</b> tab to understand it, or Search / Ask.';
    setStatus(d.stats);
    // Auto-run explain so the user immediately sees a useful overview.
    switchTab('explain');
    doExplain();
    loadSuggestions();
  } catch(e){ el('indexOut').innerHTML = '<span class="err">' + esc(e.message) + '</span>'; }
  btn.disabled = false;
}

async function loadSuggestions(){
  try {
    const d = await api('/api/suggestions');
    const askQs = d.ask_questions || d.questions || [];
    const askChips = el('suggestChips');
    if (askQs.length){
      el('suggestLabel').style.display = '';
      askChips.innerHTML = askQs.map(q =>
        '<span class="chip" style="cursor:pointer" onclick="askSuggested(this)">' + esc(q) + '</span>'
      ).join('');
    } else {
      el('suggestLabel').style.display = 'none';
      askChips.innerHTML = '';
    }

    const searchQs = d.search_queries || [];
    const searchChips = el('searchSuggestChips');
    if (searchQs.length){
      el('searchSuggestLabel').style.display = '';
      searchChips.innerHTML = searchQs.map(q =>
        '<span class="chip" style="cursor:pointer" onclick="searchSuggested(this)">' + esc(q) + '</span>'
      ).join('');
    } else {
      el('searchSuggestLabel').style.display = 'none';
      searchChips.innerHTML = '';
    }
  } catch(e){ /* suggestions are optional */ }
}

function askSuggested(elem){
  el('question').value = elem.textContent;
  doAsk();
}

function searchSuggested(elem){
  el('query').value = elem.textContent;
  doSearch();
}

function renderResults(results){
  if(!results.length) return '<div class="hint">No matches found.</div>';
  return results.map((r,i) => {
    const c = r.chunk;
    const sym = c.symbol ? ('<span class="sym">' + esc(c.kind + ' ' + c.symbol) + '</span>') : ('<span class="sym">' + esc(c.kind) + '</span>');
    return '<div class="result"><div class="head">' +
      '<span class="score">' + r.score.toFixed(3) + '</span>' +
      '<span class="loc">' + esc(c.path + ':' + c.start_line + '-' + c.end_line) + '</span>' + sym +
      '</div><pre>' + esc(c.text.replace(/\s+$/,'')) + '</pre></div>';
  }).join('');
}

async function doSearch(){
  const query = el('query').value.trim();
  if(!query) return;
  el('searchOut').innerHTML = '<span class="spinner"></span> Searching…';
  try {
    const d = await api('/api/search', { query, top_k: 8, mode: 'hybrid', semantic_weight: 0.8 });
    el('searchOut').innerHTML = renderResults(d.results);
  } catch(e){ el('searchOut').innerHTML = '<span class="err">' + esc(e.message) + '</span>'; }
}

async function doAsk(){
  const question = el('question').value.trim();
  if(!question) return;
  el('askOut').innerHTML = '<span class="spinner"></span> Thinking…';
  try {
    const d = await api('/api/ask', { question, top_k: 6 });
    let html = '<div class="card" style="margin-top:14px;background:var(--panel2)"><div class="desc">' + esc(d.answer) + '</div></div>';
    if (d.sources && d.sources.length) html += renderResults(d.sources);
    el('askOut').innerHTML = html;
  } catch(e){ el('askOut').innerHTML = '<span class="err">' + esc(e.message) + '</span>'; }
}

async function doExplain(){
  el('explainOut').innerHTML = '<span class="spinner"></span> Analysing the project…';
  try {
    const d = await api('/api/explain', {});
    let html = '<div class="card" style="margin-top:14px;background:var(--panel2)">';

    html += '<div style="font-size:18px;font-weight:700;color:var(--text)">' + esc(d.name || 'Project') + '</div>';
    if (d.project_type){
      html += '<div class="hint" style="margin-top:6px">' + esc(d.project_type) + '</div>';
    }
    const overview = (d.description || '').trim();
    if (overview){
      const paras = overview.split(/\n\n+/).filter(p => p.trim());
      html += '<h2 style="margin-top:14px">Overview</h2>';
      html += paras.map((p, i) =>
        '<div class="desc"' + (i ? ' style="margin-top:10px"' : '') + '>' + esc(p.trim()).replace(/\n/g, '<br>') + '</div>'
      ).join('');
    }

    if ((d.tech_stack||[]).length){
      html += '<h2 style="margin-top:18px">Tech stack</h2><div class="chips">' +
        d.tech_stack.map(t => '<span class="chip">' + esc(t) + '</span>').join('') + '</div>';
    }

    if ((d.architecture_layers||[]).length){
      html += '<h2 style="margin-top:18px">Architecture</h2><div class="chips">' +
        d.architecture_layers.map(([l,n]) => '<span class="chip">' + esc(l) + ' · ' + n + '</span>').join('') + '</div>';
    }

    if ((d.workflows||[]).length){
      html += '<h2 style="margin-top:18px">How it works</h2>';
      html += d.workflows.map(w => '<div class="desc" style="margin-top:6px">• ' + esc(w) + '</div>').join('');
    }

    // Quick facts.
    html += '<div class="chips">';
    html += '<span class="chip">' + d.num_files + ' files</span>';
    (d.languages||[]).slice(0,5).forEach(([l,n]) => html += '<span class="chip">' + esc(l) + ' · ' + n + '</span>');
    html += '</div>';

    // Key components with their own docstrings — the useful part.
    if ((d.components||[]).length){
      html += '<h2 style="margin-top:18px">Key components</h2>';
      d.components.slice(0,8).forEach(c => {
        html += '<div class="result"><div class="head">' +
          '<span class="sym">' + esc(c.kind + ' ' + c.symbol) + '</span>' +
          '<span class="loc">' + esc(c.location) + '</span>' +
          (c.role ? '<span class="chip" style="margin-left:8px">' + esc(c.role) + '</span>' : '') +
          '</div>';
        if (c.summary) html += '<pre>' + esc(c.summary) + '</pre>';
        html += '</div>';
      });
    } else if ((d.notable_symbols||[]).length){
      html += '<h2 style="margin-top:18px">Notable symbols</h2><div class="chips">' +
        d.notable_symbols.slice(0,10).map(s => '<span class="chip">' + esc(s) + '</span>').join('') + '</div>';
    }

    // Entry points.
    if ((d.entry_points||[]).length){
      html += '<h2 style="margin-top:18px">Entry points</h2><div class="chips">' +
        d.entry_points.map(p => '<span class="chip">' + esc(p) + '</span>').join('') + '</div>';
    }

    html += '</div>';
    el('explainOut').innerHTML = html;
  } catch(e){ el('explainOut').innerHTML = '<span class="err">' + esc(e.message) + '</span>'; }
}

async function doStats(){
  el('statsOut').innerHTML = '<span class="spinner"></span> Loading…';
  try {
    const s = await api('/api/stats');
    let html = '<div class="chips">' +
      '<span class="chip">files · ' + s.num_files + '</span>' +
      '<span class="chip">chunks · ' + s.num_chunks + '</span>' +
      '<span class="chip">embedder · ' + esc(s.embedder) + '</span>' +
      '<span class="chip">search · ' + esc(s.search_backend) + '</span>' +
      '<span class="chip">dim · ' + s.vector_dim + '</span></div>';
    html += '<h2 style="margin-top:16px">Languages</h2><div class="chips">' +
      (s.languages||[]).map(([l,n]) => '<span class="chip">' + esc(l) + ' · ' + n + '</span>').join('') + '</div>';
    html += '<h2 style="margin-top:16px">Kinds</h2><div class="chips">' +
      Object.entries(s.kinds||{}).map(([k,n]) => '<span class="chip">' + esc(k) + ' · ' + n + '</span>').join('') + '</div>';
    el('statsOut').innerHTML = html;
  } catch(e){ el('statsOut').innerHTML = '<span class="err">' + esc(e.message) + '</span>'; }
}

function switchTab(name){
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.pane === name));
  document.querySelectorAll('.pane').forEach(p => p.classList.toggle('active', p.id === 'pane-' + name));
  if (name === 'stats') doStats();
  if (name === 'explain' && !el('explainOut').innerHTML) doExplain();
  if ((name === 'ask' || name === 'search') && !el('suggestChips').innerHTML && !el('searchSuggestChips').innerHTML) loadSuggestions();
}

(async function init(){
  try {
    const d = await api('/api/status');
    if (d.loaded){ setStatus(d.stats); loadSuggestions(); }
  } catch(e){}
})();
</script>
</body>
</html>
"""
