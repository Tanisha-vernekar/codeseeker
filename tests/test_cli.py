import json
import textwrap

from codeseeker.cli import main


def _write_project(tmp_path):
    (tmp_path / "config.py").write_text(
        textwrap.dedent(
            '''\
            def load_config(path):
                """Read and parse a YAML configuration file."""
                return path
            '''
        )
    )
    (tmp_path / "db.py").write_text(
        textwrap.dedent(
            '''\
            def connect_database(url):
                """Open a database connection."""
                return url
            '''
        )
    )


def test_cli_index_then_search(tmp_path, capsys):
    _write_project(tmp_path)

    rc = main(["index", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Indexed" in out

    rc = main(["search", "parse configuration file", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "load_config" in out


def test_cli_search_json(tmp_path, capsys):
    _write_project(tmp_path)
    assert main(["index", str(tmp_path)]) == 0
    capsys.readouterr()

    rc = main(["search", "database connection", str(tmp_path), "--json", "-k", "2"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert payload
    assert payload[0]["chunk"]["symbol"] in {"connect_database", "load_config"}


def test_cli_search_without_index_errors(tmp_path, capsys):
    rc = main(["search", "anything", str(tmp_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no index found" in err


def test_cli_index_missing_path(tmp_path, capsys):
    rc = main(["index", str(tmp_path / "does-not-exist")])
    assert rc == 2
    assert "path not found" in capsys.readouterr().err


def test_cli_custom_index_dir(tmp_path, capsys):
    _write_project(tmp_path)
    index_dir = tmp_path / "custom_index"
    assert main(["index", str(tmp_path), "--index-dir", str(index_dir)]) == 0
    assert (index_dir / "meta.json").exists()
    capsys.readouterr()

    rc = main(["search", "config", str(tmp_path), "--index-dir", str(index_dir)])
    assert rc == 0
