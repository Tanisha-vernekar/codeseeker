import textwrap

from codeseeker.index import CodeIndex
from codeseeker.qa import answer_question
from codeseeker.summary import summarize_repo


def _make_project(tmp_path):
    (tmp_path / "README.md").write_text(
        "# demo\n\n"
        "A tiny demo service that loads configuration and connects to a database "
        "for integration testing purposes.\n"
    )
    (tmp_path / "config.py").write_text(
        textwrap.dedent(
            '''\
            def load_config(path):
                """Read and parse a YAML configuration file from disk."""
                return path
            '''
        )
    )
    (tmp_path / "db.py").write_text(
        textwrap.dedent(
            '''\
            class Database:
                """A thin wrapper around a database connection."""

                def connect(self, url):
                    """Open a new database connection."""
                    return url
            '''
        )
    )
    return tmp_path


def test_summarize_repo_heuristic(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root))
    summary = summarize_repo(index, root=str(root), use_llm=False)

    assert summary.num_files >= 2
    assert summary.num_chunks == len(index)
    assert not summary.llm_used
    assert summary.description
    # README first paragraph should surface in the description.
    assert "demo" in summary.description.lower()
    # Notable symbols should include the documented class.
    assert any("Database" in s for s in summary.notable_symbols)
    langs = dict(summary.languages)
    assert langs.get("python", 0) >= 2


def test_summary_json_roundtrip(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root))
    data = summarize_repo(index, root=str(root), use_llm=False).to_dict()
    assert set(["root", "num_files", "languages", "description"]).issubset(data)


def test_answer_question_heuristic(tmp_path):
    root = _make_project(tmp_path)
    index = CodeIndex.build(str(root))
    ans = answer_question(index, "how do we open a database connection?", use_llm=False)

    assert not ans.llm_used
    assert ans.sources
    # Most relevant source should relate to the database connection.
    locations = " ".join(s.chunk.symbol for s in ans.sources)
    assert "connect" in locations.lower() or "Database" in locations
    assert "db.py" in ans.answer


def test_answer_question_no_results(tmp_path):
    (tmp_path / "notes.txt").write_text("nothing indexable")
    index = CodeIndex.build(str(tmp_path))
    ans = answer_question(index, "anything", use_llm=False)
    assert ans.sources == []
    assert "No relevant code" in ans.answer
