# codeseeker

Semantic code search over your local codebase — from the command line.

`codeseeker` indexes a project by breaking it into meaningful chunks
(functions, classes, methods, and file blocks) and embedding them into vectors.
You can then search with natural language ("where do we validate a JWT?") or
code-like queries and get back the most relevant snippets, ranked by semantic
similarity.

The default embedding backend is a pure-NumPy **TF-IDF** model with
identifier-aware tokenisation, so `codeseeker` works **completely offline** with
no model downloads. An optional `sentence-transformers` backend is available for
dense neural embeddings when you want them.

## Features

- **AST-aware chunking** for Python (functions/classes/methods keep their names
  and docstrings); a sliding-window fallback handles every other language.
- **Identifier-aware tokenisation** — `getUserById` and `get_user_by_id` both
  match a query like *"fetch user by id"*.
- **Offline by default** — no network, no API keys, no model downloads.
- **Pluggable backends** — swap in `sentence-transformers` with one flag.
- **Portable on-disk index** stored in a `.codeseeker/` directory.
- **Simple CLI** plus a small, importable Python API.

## Installation

```bash
# From a clone of this repository:
pip install .

# With the optional neural backend:
pip install ".[transformers]"
```

Or just run it in place (only NumPy is required for the default backend):

```bash
pip install numpy
python -m codeseeker --help
```

## Usage

### 1. Build an index

```bash
codeseeker index path/to/project
# or index the current directory
codeseeker index
```

Useful options:

```bash
# Only index certain extensions
codeseeker index . --ext .py,.js,.ts

# Exclude extra directories (added to sensible defaults)
codeseeker index . --exclude migrations,vendor

# Use the neural backend (requires the 'transformers' extra)
codeseeker index . --backend sentence-transformers --model all-MiniLM-L6-v2
```

The index is written to `<project>/.codeseeker/` by default (override with
`--index-dir`).

### 2. Search

```bash
codeseeker search "parse a configuration file"
codeseeker search "retry an http request with backoff" -k 10
codeseeker search "where do we hash passwords" --json
```

Example output:

```
Top 3 results for: parse a configuration file

[1] 0.612  src/config.py:12-34  function load_config
    def load_config(path):
        """Read and validate a YAML config file."""
        ...
```

## Python API

```python
from codeseeker import CodeIndex

index = CodeIndex.build("path/to/project")
index.save("path/to/project/.codeseeker")

# later ...
index = CodeIndex.load("path/to/project/.codeseeker")
for hit in index.search("open a database connection", top_k=5):
    print(hit.score, hit.chunk.location, hit.chunk.symbol)
```

## How it works

1. **Chunking** (`codeseeker.chunking`): Python files are parsed with `ast`
   into function/class/method chunks that retain their qualified name and
   docstring. Other files are split into overlapping line windows.
2. **Embedding** (`codeseeker.embeddings`): each chunk is tokenised
   (identifiers are split into sub-words) and turned into an L2-normalised
   TF-IDF vector. Symbol names and docstrings are up-weighted.
3. **Search** (`codeseeker.index`): the query is embedded the same way and
   cosine similarity (a dot product on normalised vectors) ranks the chunks.

## Development

```bash
pip install ".[dev]"
pytest
```

## License

MIT
