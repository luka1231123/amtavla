import json

from brain.action_runner import ActionRunner
from brain.contracts import Action, ActionType
from tools.fileparse import FileParseClient
from tools.webfetch import WebFetchClient, html_to_text


_HTML = (
    "<html><head><title>Pope Essay</title></head><body>"
    "<script>var x = 1;</script><h1>An Essay on Criticism</h1>"
    "<p>To err is human; to forgive, divine.</p></body></html>"
)


def _html_fetcher(url, timeout):
    return "text/html; charset=utf-8", _HTML.encode("utf-8")


def test_html_to_text_strips_scripts_and_tags():
    text = html_to_text(_HTML)
    assert "var x = 1" not in text
    assert "An Essay on Criticism" in text
    assert "To err is human" in text


def test_web_fetch_extracts_title_text_and_citation():
    client = WebFetchClient(fetcher=_html_fetcher)
    result = client.fetch("https://example.com/pope")
    assert result["title"] == "Pope Essay"
    assert "To err is human" in result["content"]
    assert result["source_id"].startswith("web:")


def test_web_fetch_rejects_non_url():
    client = WebFetchClient(fetcher=_html_fetcher)
    result = client.fetch("look it up")
    assert "valid http(s) URL" in result["error"]


def test_web_fetch_network_error_is_structured():
    def _boom(url, timeout):
        raise OSError("connection refused")

    result = WebFetchClient(fetcher=_boom).fetch("https://example.com")
    assert "Fetch failed" in result["error"]


def test_action_runner_web_fetch_produces_web_source():
    runner = ActionRunner(web_fetch_client=WebFetchClient(fetcher=_html_fetcher))
    action = Action.create(ActionType.WEB_FETCH, "https://example.com/pope")
    result = runner.run(action, user_input="read https://example.com/pope")
    assert result.ok is True
    assert result.sources[0].kind == "web"
    assert result.sources[0].source_id.startswith("web:")


def test_file_parse_json(tmp_path):
    (tmp_path / "data.json").write_text('{"b": 2, "a": 1}')
    result = FileParseClient(root=tmp_path).parse("data.json")
    assert result["kind"] == "json"
    assert '"a": 1' in result["content"]


def test_file_parse_csv(tmp_path):
    (tmp_path / "rows.csv").write_text("name,age\nMira,30\nLuka,28\n")
    result = FileParseClient(root=tmp_path).parse("rows.csv")
    assert "name | age" in result["content"]
    assert "Mira | 30" in result["content"]


def test_file_parse_path_escape_refused(tmp_path):
    result = FileParseClient(root=tmp_path).parse("../secret.json")
    assert "outside allowed root" in result["error"]


def test_file_parse_pdf_without_dependency_is_graceful(tmp_path):
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
    result = FileParseClient(root=tmp_path).parse("doc.pdf")
    # No pypdf installed in the base env → a clear instruction, not a crash.
    assert "error" in result
    assert "pypdf" in result["error"]


def test_action_runner_file_parse(tmp_path):
    (tmp_path / "d.json").write_text('{"ok": true}')
    runner = ActionRunner(file_parse_client=FileParseClient(root=tmp_path))
    action = Action.create(ActionType.FILE_PARSE, "d.json")
    result = runner.run(action, user_input="parse d.json")
    assert result.ok is True
    assert result.sources[0].metadata["tier"] == "T0"
