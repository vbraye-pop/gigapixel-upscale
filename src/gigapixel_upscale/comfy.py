"""Minimal client for one remote ComfyUI server.

Built for a RunPod HTTPS proxy, which resets connections now and then and serves an HTML page during
hiccups: every read treats a failed or non-JSON reply as a transient miss, and the prompt id is chosen
client-side so a submit that died mid-flight can be checked before it is sent again.
"""

import contextlib
import json
import mimetypes
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import requests

# the RunPod proxy rejects some default library user agents
HEADERS = {"User-Agent": "Mozilla/5.0"}
DONE, FAILED = {"completed"}, {"failed", "cancelled"}


class ComfyError(RuntimeError):
    pass


def format_node_errors(payload: dict, graph: dict) -> str:
    """One line per failing input, named by node title, from a /prompt validation reply."""
    lines = [payload.get("error", {}).get("message", "prompt rejected")]
    for nid, err in payload.get("node_errors", {}).items():
        title = graph.get(nid, {}).get("_meta", {}).get("title", err.get("class_type", nid))
        for e in err.get("errors", []):
            extra = e.get("extra_info", {})
            where = f" (input {extra['input_name']})" if "input_name" in extra else ""
            lines.append(f"  [{title}] {e.get('message', '')}: {e.get('details', '')}{where}")
    return "\n".join(lines)


class Comfy:
    def __init__(self, url: str, timeout: int = 60, headers: dict | None = None):
        if not url:
            raise ComfyError("no ComfyUI server: pass --server or set COMFY_URL")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.headers = {**HEADERS, **(headers or {})}

    def _get_json(self, path: str) -> dict | None:
        try:
            r = requests.get(f"{self.url}{path}", headers=self.headers, timeout=self.timeout)
            if r.status_code != 200:
                return None
            # history replies can carry raw control characters inside node text
            return json.loads(r.text, strict=False)
        except (requests.RequestException, ValueError):
            return None

    def upload(self, path: Path) -> str:
        # a unique name, so two people on a shared server never overwrite each other's input
        name = f"{path.stem}_{uuid.uuid4().hex[:8]}{path.suffix}"
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with path.open("rb") as f:
            r = requests.post(
                f"{self.url}/upload/image",
                headers=self.headers,
                files={"image": (name, f, mime)},
                data={"overwrite": "true"},
                timeout=max(self.timeout, 600),
            )
        if r.status_code != 200:
            raise ComfyError(f"upload failed ({r.status_code}): {r.text[:300]}")
        body = r.json()
        return f"{body['subfolder']}/{body['name']}" if body.get("subfolder") else body["name"]

    def job_status(self, prompt_id: str) -> str | None:
        job = self._get_json(f"/api/jobs/{prompt_id}")
        return job.get("status") if job else None

    def submit(self, graph: dict) -> str:
        prompt_id = str(uuid.uuid4())
        for _ in range(2):
            try:
                r = requests.post(
                    f"{self.url}/prompt",
                    headers=self.headers,
                    json={"prompt": graph, "prompt_id": prompt_id},
                    timeout=self.timeout,
                )
            except requests.RequestException:
                # the request may have reached the server before the connection dropped
                if self.job_status(prompt_id):
                    return prompt_id
                continue
            if r.status_code == 200:
                return prompt_id
            try:
                raise ComfyError(format_node_errors(r.json(), graph))
            except ValueError:
                raise ComfyError(f"queue failed ({r.status_code}): {r.text[:600]}") from None
        raise ComfyError("could not reach the server to queue the prompt")

    def wait(
        self,
        prompt_id: str,
        on_status: Callable[[str, float], None] | None = None,
        poll: float = 3.0,
        timeout: float = 7200,
    ) -> dict:
        """Block until the prompt finishes and return its outputs, keyed by node id."""
        start = time.time()
        while time.time() - start < timeout:
            status = self.job_status(prompt_id)
            if on_status and status:
                on_status(status, time.time() - start)
            if status in DONE | FAILED:
                entry = (self._get_json(f"/history/{prompt_id}") or {}).get(prompt_id)
                if entry and status in FAILED:
                    raise ComfyError(f"{status}: {_execution_error(entry)}")
                if entry:
                    return entry["outputs"]
            time.sleep(poll)
        raise ComfyError(f"timed out after {timeout:.0f}s waiting on {prompt_id}")

    def cancel(self, prompt_id: str) -> None:
        with contextlib.suppress(requests.RequestException):
            requests.post(f"{self.url}/api/jobs/{prompt_id}/cancel", headers=self.headers, timeout=self.timeout)

    def download(self, ref: dict, dest: Path) -> Path:
        """Stream one output image to dest, retrying the whole file once if the proxy drops it."""
        params = {k: ref[k] for k in ("filename", "subfolder", "type")}
        part = dest.with_name(dest.name + ".part")
        for attempt in range(2):
            try:
                with requests.get(
                    f"{self.url}/view", headers=self.headers, params=params, stream=True, timeout=max(self.timeout, 600)
                ) as r:
                    r.raise_for_status()
                    expected = int(r.headers.get("Content-Length", 0))
                    with part.open("wb") as f:
                        for chunk in r.iter_content(8 << 20):
                            f.write(chunk)
                if expected and part.stat().st_size != expected:
                    raise requests.ConnectionError("truncated download")
                return part.replace(dest)
            except requests.RequestException as e:
                if attempt:
                    raise ComfyError(f"download of {ref['filename']} failed: {e}") from e
        return dest


def _execution_error(entry: dict) -> str:
    for kind, data in entry.get("status", {}).get("messages", []):
        if kind == "execution_error":
            return f"{data.get('node_type')}: {data.get('exception_message', '').strip()}"
    return "no error message in history"
