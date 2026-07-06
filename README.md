# codeseeker

**LLM-powered semantic code search over local and remote repositories.**

`codeseeker` lets you point at a codebase — a local folder *or* a remote GitHub
repo — and search it with natural language ("where do we validate a JWT?"),
get a short explanation of what the project does, and ask grounded questions
about the code. It parses code with an AST-aware pipeline, embeds
function/class-level chunks into vectors, and serves fast similarity search via
**FAISS** (with an exact NumPy fallback).

It runs **completely offline by default**: the default embedding backend is a
pure-NumPy TF-IDF model with identifier-aware tokenisation, so there are no
model downloads or API keys required. Neural embeddings
(`sentence-transformers`) and LLM answers (any OpenAI-compatible API) are
optional, drop-in upgrades.

## Features

- **Search anything, from anywhere** — index a local path, a git URL, or a
  `owner/repo` shorthand. Remote repos are **pulled to your machine** (shallow
  clone, cached) and then indexed.
- **Natural-language semantic search** — identifier-aware so `getUserById`
  matches *"fetch user by id"*. Filter by language, symbol kind, or path.
- **Project explanation** — `codeseeker explain` gives a short, readable summary
  of what a project does (README + language mix + notable symbols; optionally
  polished by an LLM).
- **Ask questions (RAG)** — `codeseeker ask "how does retry work?"` retrieves the
  most relevant code and, if an LLM is configured, synthesises a cited answer.
- **FAISS-backed vector search** — exact (`IndexFlatIP`) or approximate
  (`IndexIVFFlat`) for sub-100ms queries at scale; automatic NumPy fallback.
- **AST-aware chunking** — Python functions/classes/methods keep their qualified
  names + docstrings; a sliding-window fallback covers every other language.
- **Interactive mode, stats, and JSON output** for scripting and dashboards.
- **Pluggable & portable** — the on-disk `.codeseeker/` index works regardless
  of which optional backends are installed.

## Web UI — run one file

Prefer a browser instead of the terminal? Launch the web app and everything
(index, search, ask, explain, stats) happens in a single page:

```bash
pip install flask numpy      # minimal deps for the web UI
python app.py                # starts a local server and opens your browser
```

Then, in the page: type a repo (a local path like `.`, or `owner/repo`, or a
GitHub URL), click **Index**, and use the **Search / Ask / Explain / Stats**
tabs. Equivalent to `codeseeker web` if the package is installed.

![web UI](docs/web-ui.svg)

## Installation

```bash
# Core (offline TF-IDF backend, only needs NumPy):
pip install .

# Web UI:
pip install ".[web]"

# Fast search at scale:
pip install ".[faiss]"

# Neural embeddings:
pip install ".[transformers]"

# LLM-powered explanations & answers:
pip install ".[llm]"

# Everything:
pip install ".[all]"
```

## Quick start

```bash
# 1. Index a local project ...
codeseeker index path/to/project

# ... or pull & index a remote repo (cloned to ~/.cache/codeseeker/repos):
codeseeker index benjaminp/six
codeseeker index https://github.com/pallets/flask

# 2. Search it
codeseeker search "parse a configuration file" -k 5

# 3. Understand it
codeseeker explain

# 4. Ask about it
codeseeker ask "how are remote repositories cloned to the local machine?"

# 5. Inspect the index
codeseeker stats
```

### What does \"index\" mean?

When you run **index**, codeseeker does two things:

1. If the source is remote (`owner/repo` or URL), it **downloads/clones that repo to your local machine**.
2. It reads code files, splits them into meaningful chunks (functions/classes/blocks), and stores vector embeddings in a local `.codeseeker/` folder.

After this one-time step, `search`, `ask`, and `explain` become very fast because they query the local index instead of scanning files from scratch each time.

## Commands

### `index` — build a semantic index

```bash
codeseeker index <source> [options]
```

`<source>` is a local path, a git URL, or an `owner/repo` shorthand.

| Option | Description |
| --- | --- |
| `--index-dir DIR` | Where to store the index (default: `<source>/.codeseeker`). |
| `--ext .py,.js` | Restrict to specific file extensions. |
| `--exclude a,b` | Extra directory names to skip (added to sensible defaults). |
| `--backend` | `tfidf` (default, offline) or `sentence-transformers`. |
| `--model NAME` | Model for the neural backend (default `all-MiniLM-L6-v2`). |
| `--faiss auto\|on\|off` | Use FAISS for search (`auto` = when installed). |
| `--depth N` / `--branch B` / `--update` | Clone controls for remote repos. |
| `--cache-dir DIR` | Where cloned repos are cached. |

### `clone` — just pull a repo to the local machine

```bash
codeseeker clone owner/repo
codeseeker clone https://github.com/owner/repo --update
```

### `search` — semantic search

```bash
codeseeker search "retry an http request with backoff" -k 10
codeseeker search "hash a password" --lang python --kind function
codeseeker search "database migration" --path migrations/
codeseeker search --interactive        # REPL mode
codeseeker search "open db connection" --json
```

### `explain` — short project summary

```bash
codeseeker explain            # heuristic summary (offline)
codeseeker explain --llm on   # polished by an LLM (needs OPENAI_API_KEY + [llm])
codeseeker explain --json
```

### `ask` — grounded Q&A (RAG)

```bash
codeseeker ask "where is authentication handled?"
codeseeker ask "how does the cache work?" --json
```

Without an LLM, `ask` returns the most relevant, cited code locations. With an
LLM configured (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `pip install
".[llm]"`), it synthesises an answer that cites its sources.

### `stats` — index overview

```bash
codeseeker stats            # language/kind breakdown, backend, dimensions
codeseeker stats --json
```

### `web` — browser UI

```bash
codeseeker web                       # opens http://127.0.0.1:8000 in your browser
codeseeker web --port 8080 --no-browser
python app.py                        # equivalent one-file launcher
```

## Python API

```python
from codeseeker import CodeIndex, resolve_source, summarize_repo, answer_question

# Pull a remote repo locally, then index it
repo = resolve_source("benjaminp/six")
index = CodeIndex.build(repo.local_path, origin=repo.origin, is_remote=True)
index.save(repo.local_path + "/.codeseeker")

# Semantic search (with filters)
for hit in index.search("open a database connection", top_k=5, kinds=["function"]):
    print(hit.score, hit.chunk.location, hit.chunk.symbol)

# Explain the project
print(summarize_repo(index).description)

# Ask a grounded question
print(answer_question(index, "how is retry implemented?").answer)
```

## How it works

1. **Fetch** (`codeseeker.repo`): local paths are used as-is; remote sources are
   shallow-cloned into a cache directory.
2. **Chunk** (`codeseeker.chunking`): Python is parsed with `ast` into
   function/class/method chunks (keeping names + docstrings); other files use an
   overlapping line-window chunker.
3. **Embed** (`codeseeker.embeddings`): each chunk becomes an L2-normalised
   vector. The default TF-IDF backend splits identifiers into sub-words and
   up-weights symbol names and docstrings.
4. **Index** (`codeseeker.vectorstore`): vectors are served by FAISS
   (`IndexFlatIP`/`IndexIVFFlat`) or an exact NumPy store.
5. **Search / Explain / Ask** (`codeseeker.index`, `.summary`, `.qa`): the query
   is embedded the same way; cosine similarity ranks chunks, which also feed the
   summary and RAG answering.

## Configuration for optional LLM features

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional (for compatible APIs)
pip install ".[llm]"
```

## Development

```bash
pip install ".[dev]"
pytest
```

## License

MIT
