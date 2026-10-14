import json

import pytest
import requests

from gigapixel_upscale import comfy
from gigapixel_upscale.comfy import Comfy, ComfyError

URL = "https://pod.example"


class Reply:
    def __init__(self, status=200, body=None, text=None, chunks=(), headers=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body)
        self._chunks = chunks
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, _size):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(comfy.time, "sleep", lambda _: None)


def test_submit_sends_a_client_chosen_prompt_id_and_returns_it(monkeypatch):
    sent = {}

    def post(url, json=None, **_):
        sent.update(json)
        return Reply(body={"prompt_id": json["prompt_id"], "number": 1})

    monkeypatch.setattr(comfy.requests, "post", post)
    prompt_id = Comfy(URL).submit({"1": {"class_type": "LoadImage", "inputs": {}}})
    assert prompt_id == sent["prompt_id"]
    assert sent["prompt"]["1"]["class_type"] == "LoadImage"


def test_submit_reports_validation_errors_by_node_title(monkeypatch):
    reply = {
        "error": {"message": "Prompt outputs failed validation"},
        "node_errors": {
            "2": {
                "class_type": "UpscaleModelLoader",
                "errors": [
                    {
                        "message": "Value not in list",
                        "details": "model_name: 'x.pth' not in []",
                        "extra_info": {"input_name": "model_name"},
                    }
                ],
            }
        },
    }
    monkeypatch.setattr(comfy.requests, "post", lambda *a, **k: Reply(400, reply))
    graph = {"2": {"class_type": "UpscaleModelLoader", "_meta": {"title": "Nomos2 loader"}, "inputs": {}}}
    with pytest.raises(ComfyError, match=r"\[Nomos2 loader\] Value not in list.*\(input model_name\)"):
        Comfy(URL).submit(graph)


def test_a_submit_that_reached_the_server_before_the_connection_dropped_is_not_sent_twice(monkeypatch):
    posts = []

    def post(*_, **__):
        posts.append(1)
        raise requests.ConnectionError("reset by peer")

    monkeypatch.setattr(comfy.requests, "post", post)
    monkeypatch.setattr(comfy.requests, "get", lambda *a, **k: Reply(body={"status": "pending"}))
    Comfy(URL).submit({})
    assert len(posts) == 1


def test_wait_rides_out_proxy_html_pages_and_returns_the_outputs(monkeypatch, no_sleep):
    replies = iter(
        [
            Reply(text="<html>502 bad gateway</html>"),
            Reply(body={"status": "in_progress"}),
            Reply(body={"status": "completed"}),
        ]
    )
    outputs = {"20": {"images": [{"filename": "master_00001_.png", "subfolder": "", "type": "output"}]}}

    def get(url, **_):
        if "/history/" in url:
            return Reply(body={"abc": {"outputs": outputs, "status": {"status_str": "success"}}})
        return next(replies)

    seen = []
    monkeypatch.setattr(comfy.requests, "get", get)
    assert Comfy(URL).wait("abc", on_status=lambda s, _: seen.append(s)) == outputs
    assert seen == ["in_progress", "completed"]


def test_wait_raises_the_node_error_when_the_job_failed(monkeypatch, no_sleep):
    history = {
        "abc": {
            "outputs": {},
            "status": {
                "status_str": "error",
                "messages": [["execution_error", {"node_type": "KSampler", "exception_message": "CUDA out of memory"}]],
            },
        }
    }

    def get(url, **_):
        return Reply(body=history) if "/history/" in url else Reply(body={"status": "failed"})

    monkeypatch.setattr(comfy.requests, "get", get)
    with pytest.raises(ComfyError, match="KSampler: CUDA out of memory"):
        Comfy(URL).wait("abc")


def test_download_writes_the_file_and_leaves_no_partial_behind(monkeypatch, tmp_path):
    monkeypatch.setattr(
        comfy.requests, "get", lambda *a, **k: Reply(chunks=[b"abc", b"def"], headers={"Content-Length": "6"})
    )
    dest = Comfy(URL).download({"filename": "m.png", "subfolder": "", "type": "output"}, tmp_path / "m.png")
    assert dest.read_bytes() == b"abcdef"
    assert list(tmp_path.iterdir()) == [dest]


def test_download_retries_once_then_fails_on_a_truncated_file(monkeypatch, tmp_path):
    calls = []

    def get(*_, **__):
        calls.append(1)
        return Reply(chunks=[b"abc"], headers={"Content-Length": "6"})

    monkeypatch.setattr(comfy.requests, "get", get)
    with pytest.raises(ComfyError, match="truncated"):
        Comfy(URL).download({"filename": "m.png", "subfolder": "", "type": "output"}, tmp_path / "m.png")
    assert len(calls) == 2


def test_cancel_targets_the_job(monkeypatch):
    urls = []
    monkeypatch.setattr(comfy.requests, "post", lambda url, **_: urls.append(url) or Reply(body={"cancelled": True}))
    Comfy(URL).cancel("abc")
    assert urls == [f"{URL}/api/jobs/abc/cancel"]


def test_a_missing_server_url_is_reported_plainly():
    with pytest.raises(ComfyError, match="COMFY_URL"):
        Comfy("")
