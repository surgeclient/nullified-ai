"""Run tools/mod_checks.py on GitHub Actions (check-mod workflow) and bring the results back. Standard library only.

    gh = GitHubChecks("owner/repo", token)
    results = gh.run({"items": [{"id": "req0", "files": {...}}]})    # -> {"checks": [...], "package": None}
    results = gh.run({"items": [...], "package": True}, save_to=Path("output"))  # also saves the .jar/.zip

Token: a fine-grained GitHub token for this repo with "Actions: Read and write" + "Contents: Read".
"""
import base64
import gzip
import io
import json
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

WORKFLOW = "check-mod.yml"
MAX_PAYLOAD = 60000  # workflow_dispatch inputs are capped at 65,535 characters in total


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class GitHubChecks:
    def __init__(self, repo: str, token: str, ref: str = ""):
        self.repo, self.token = repo, token
        self.ref = ref or self._api(f"/repos/{repo}")["default_branch"]

    def _api(self, path: str, body: dict | None = None, raw: bool = False):
        req = urllib.request.Request(f"https://api.github.com{path}",
                                     data=json.dumps(body).encode() if body is not None else None,
                                     method="POST" if body is not None else "GET",
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Accept": "application/vnd.github+json",
                                              "X-GitHub-Api-Version": "2022-11-28"})
        try:
            if raw:  # artifact downloads redirect to storage that must not receive the GitHub token
                try:
                    urllib.request.build_opener(_NoRedirect).open(req, timeout=60)
                except urllib.error.HTTPError as e:
                    if e.code in (301, 302, 303, 307, 308):
                        with urllib.request.urlopen(e.headers["Location"], timeout=300) as resp:
                            return resp.read()
                    raise
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                return json.loads(data) if data else None
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="ignore")[:500]
            hint = {401: "the token is wrong or expired",
                    403: "the token needs 'Actions: Read and write' + 'Contents: Read' on this repo",
                    404: f"repo {self.repo} not found, the token can't see it, or {WORKFLOW} isn't on the "
                         "default branch yet"}.get(e.code, "")
            raise SystemExit(f"GitHub API {e.code} on {path}: {hint}\n{detail}")

    @staticmethod
    def encode(job: dict) -> str:
        return base64.b64encode(gzip.compress(json.dumps(job).encode())).decode()

    def split(self, items: list[dict]) -> list[list[dict]]:
        """Group mods into as few runs as fit the dispatch size limit (fewer runs = fewer Actions minutes)."""
        groups, current = [], []
        for item in items:
            if current and len(self.encode({"items": current + [item]})) > MAX_PAYLOAD:
                groups.append(current)
                current = []
            current.append(item)
        if current:
            groups.append(current)
        for g in groups:
            if len(self.encode({"items": g})) > MAX_PAYLOAD:
                raise SystemExit(f"mod {g[0]['id']} is too large to send to GitHub Actions in one go")
        return groups

    def run(self, job: dict, save_to: Path | None = None, poll: int = 10) -> dict:
        job_id = uuid.uuid4().hex[:12]
        wf = f"/repos/{self.repo}/actions/workflows/{WORKFLOW}"
        self._api(f"{wf}/dispatches", {"ref": self.ref, "inputs": {"job_id": job_id, "payload": self.encode(job)}})

        run, started = None, time.time()
        while True:
            time.sleep(poll)
            if run is None:
                runs = self._api(f"{wf}/runs?event=workflow_dispatch&per_page=30")["workflow_runs"]
                run = next((r for r in runs if job_id in (r.get("display_title") or "")), None)
                if run is None:
                    if time.time() - started > 300:
                        raise SystemExit("GitHub never started the check-mod run (is Actions enabled on the repo?)")
                    continue
                print(f"    GitHub Actions run: {run['html_url']}", flush=True)
            run = self._api(f"/repos/{self.repo}/actions/runs/{run['id']}")
            if run["status"] == "completed":
                break

        arts = self._api(f"/repos/{self.repo}/actions/runs/{run['id']}/artifacts")["artifacts"]
        art = next((a for a in arts if a["name"] == "results"), None)
        if art is None:
            raise SystemExit(f"check-mod run ended '{run['conclusion']}' without results: {run['html_url']}")
        archive = zipfile.ZipFile(io.BytesIO(self._api(art["archive_download_url"].replace(
            "https://api.github.com", ""), raw=True)))
        if "results.json" not in archive.namelist():
            raise SystemExit(f"check-mod run ended '{run['conclusion']}' without results.json: {run['html_url']}")
        results = json.loads(archive.read("results.json"))
        pkg = results.get("package")
        if pkg and save_to is not None:
            folder = save_to / pkg["mod_id"]
            folder.mkdir(parents=True, exist_ok=True)
            for name in (pkg["jar"], pkg["zip"]):
                (folder / name).write_bytes(archive.read(name))
            pkg["folder"] = str(folder.resolve())
        return results
