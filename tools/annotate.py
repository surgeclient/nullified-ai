"""Publish a run's result as a GitHub Actions notice, so the web page (docs/index.html) can read it through
the API: the notice titled "nullified-result" / "nullified-teach" carries the result as JSON.

    python tools/annotate.py build output/result.json
    python tools/annotate.py teach output/data/<run>.summary.json
"""
import json
import sys
from pathlib import Path


def notice(title: str, data: dict) -> None:
    text = json.dumps(data, ensure_ascii=False)
    # workflow-command escaping: % first, then line breaks
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::notice title={title}::{text}")


def main():
    kind, path = sys.argv[1], Path(sys.argv[2])
    if not path.exists():
        notice(f"nullified-{'result' if kind == 'build' else 'teach'}", {"crashed": True})
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if kind == "build":
        jars = sorted(p.name for p in Path("output").glob("*/*.jar"))
        notice("nullified-result", {
            "ok": data.get("ok"), "refused": data.get("refused"), "request": data.get("request"),
            "plan": (data.get("plan") or "")[:3000], "stages": data.get("stage_history"),
            "errors": [e[:400] for e in (data.get("errors") or [])[:5]] if not data.get("ok") else [],
            "jar": jars[0] if jars else None, "files": sorted(data.get("files") or {})[:60]})
    else:
        notice("nullified-teach", data)


if __name__ == "__main__":
    main()
