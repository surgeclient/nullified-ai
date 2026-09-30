"""Check and package generated mods with plain Java + Gradle (no Modal). Runs on GitHub Actions or your PC.

  check(id, files)        -> {"ok", "stage", "errors"}   compile, boot a real 1.21.11 server, validate assets
  package(files, out_dir) -> {"mod_id", "jar", "zip", "placeholder_textures"}

CLI (what .github/workflows/check-mod.yml runs):
    python tools/mod_checks.py --job job.json --out results
job.json: {"items": [{"id", "files"}], "package": false} (package=true builds items[0] without re-checking). Writes results/results.json (+ jar/zip when packaging).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_compile import GRADLEW, check_sample, prepare_work_project  # noqa: E402
from build_mod import assemble_project, read_mod_id  # noqa: E402
from runtime_check import run_server, validate_assets  # noqa: E402
from textures import placeholder_textures  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VERSION = "1.21.11"
WORK = ROOT / ".work"
PROBE_JAR = WORK / "probe.jar"


def gradle(args: list[str], cwd: Path) -> None:
    cmd = [GRADLEW, *args, "--console=plain"]
    subprocess.run(["cmd", "/c", *cmd] if os.name == "nt" else cmd, cwd=cwd, check=True)


def ensure_probe() -> Path:
    """Build the runtime probe mod once; it records what the mod registered and stops the test server."""
    if PROBE_JAR.exists():
        return PROBE_JAR
    print("Building the probe mod (first run only)...", flush=True)
    build = WORK / "probe_build"
    shutil.rmtree(build, ignore_errors=True)
    shutil.copytree(ROOT / "templates" / VERSION, build, ignore=shutil.ignore_patterns("src"))
    shutil.copytree(ROOT / "tools" / "probe" / "src", build / "src")
    if os.name != "nt":
        (build / "gradlew").chmod(0o755)
    gradle(["build"], build)
    jar = next(j for j in (build / "build" / "libs").glob("*.jar") if not j.name.endswith("-sources.jar"))
    shutil.copy2(jar, PROBE_JAR)
    return PROBE_JAR


def write_sample(folder: Path, files: dict[str, str]) -> None:
    shutil.rmtree(folder, ignore_errors=True)
    for rel, body in files.items():
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_text(body, encoding="utf-8")


def check(item_id: str, files: dict[str, str]) -> dict:
    """Same stages as the Modal compile workers: restrictions/json/compile -> runtime -> assets -> passed."""
    start = time.time()
    work = prepare_work_project(VERSION)
    sample = WORK / "samples" / item_id
    write_sample(sample, files)
    r = check_sample(sample, work, timeout=300)
    ok, stage, errors = r["ok"], r["stage"], r["errors"]
    if ok:
        ok, errors, probe = run_server(work, ensure_probe(), gradlew=GRADLEW)
        stage = "runtime"
        if ok:
            errors = validate_assets(files, probe)
            ok, stage = not errors, "assets" if errors else "passed"
    return {"id": item_id, "ok": ok, "stage": stage, "errors": errors, "seconds": round(time.time() - start, 1)}


def package(files: dict[str, str], out_dir: Path) -> dict:
    """Full Gradle build -> <out_dir>/<modid>-1.0.0.jar and <modid>-project.zip (a clean project for an IDE)."""
    sample = WORK / "package"
    write_sample(sample, files)
    placeholders = placeholder_textures(files)
    for rel, png in placeholders.items():
        (sample / rel).parent.mkdir(parents=True, exist_ok=True)
        (sample / rel).write_bytes(png)
    mod_id = read_mod_id(sample / "src")
    project = WORK / "build" / mod_id
    assemble_project(sample, VERSION, project, mod_id)
    gradle(["build"], project)
    out_dir.mkdir(parents=True, exist_ok=True)
    jar = next(j for j in (project / "build" / "libs").glob("*.jar") if not j.name.endswith("-sources.jar"))
    jar_name = f"{mod_id}-1.0.0.jar"
    shutil.copy2(jar, out_dir / jar_name)
    clean = WORK / "zip" / mod_id
    shutil.rmtree(clean, ignore_errors=True)
    shutil.copytree(project, clean, ignore=shutil.ignore_patterns("build", ".gradle", "run"))
    zip_name = Path(shutil.make_archive(str(out_dir / f"{mod_id}-project"), "zip", clean)).name
    return {"mod_id": mod_id, "jar": jar_name, "zip": zip_name, "placeholder_textures": sorted(placeholders)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = {"checks": [], "package": None}
    for item in [] if job.get("package") else job["items"]:  # packaging jobs were already checked
        try:
            res = check(item["id"], item["files"])
        except Exception as e:  # one broken mod must not lose the results of the others
            res = {"id": item["id"], "ok": False, "stage": "crash", "errors": [repr(e)], "seconds": 0}
        results["checks"].append(res)
        print(f"{res['id']}: {'PASS' if res['ok'] else 'FAIL'} ({res['stage']}, {res['seconds']}s)", flush=True)
        for e in res["errors"][:3]:
            print("    " + e.replace("\n", "\n    "), flush=True)
    if job.get("package"):
        results["package"] = package(job["items"][0]["files"], out)
    (out / "results.json").write_text(json.dumps(results), encoding="utf-8")


if __name__ == "__main__":
    main()
