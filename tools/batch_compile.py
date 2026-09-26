"""Compile-check many mod samples against one template, reusing a single warm Gradle setup.

A sample is a folder containing a `src/` tree (same layout as the template's src/).
Each sample's src/ is swapped into the work project and compiled. Setup (downloading and
remapping Minecraft) only happens once, so each extra sample takes seconds instead of minutes.

Usage:
    python tools/batch_compile.py --samples samples/smoke --template 1.21.11 --out results/smoke.jsonl
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from restrictions import scan_dir  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GRADLEW = "gradlew.bat" if os.name == "nt" else "./gradlew"
COMPILE_TASKS = ["compileJava", "compileClientJava", "processResources", "processClientResources"]
ERROR_LINE = re.compile(r"\.java:\d+: error:")


def prepare_work_project(template: str) -> Path:
    """Copy the template (minus its src/) into .work/, keeping any existing Gradle state."""
    src = ROOT / "templates" / template
    work = ROOT / ".work" / template
    work.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name == "src":
            continue
        dest = work / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    if os.name != "nt":
        (work / "gradlew").chmod(0o755)
    return work


def invalid_json_files(src: Path) -> list[str]:
    bad = []
    for file in src.rglob("*.json"):
        text = file.read_text(encoding="utf-8", errors="ignore")
        # fabric.mod.json contains ${version}, which is valid JSON inside a string.
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            bad.append(f"{file.relative_to(src)}: {e}")
    return bad


def extract_errors(output: str, limit: int = 20) -> list[str]:
    """Pull javac errors out of Gradle output: the error line, the code, and any symbol/location lines."""
    lines = output.splitlines()
    errors = []
    for i, line in enumerate(lines):
        if ERROR_LINE.search(line):
            block = [line]
            for follow in lines[i + 1:i + 7]:
                if ERROR_LINE.search(follow) or follow.startswith(("> Task", "FAILURE", "BUILD")) or re.match(r"^\d+ errors?$", follow.strip()):
                    break
                block.append(follow)
            errors.append("\n".join(block).strip())
    # Gradle repeats every javac error in its failure summary (indented) - keep one copy of each.
    unique = {}
    for e in errors:
        unique.setdefault(" ".join(e.split()), e)
    errors = list(unique.values())
    if not errors:
        # Not a javac error (bad fabric.mod.json expansion, Gradle failure...). Keep the tail.
        errors = ["\n".join(line for line in lines if line.strip())[-2000:]]
    return errors[:limit]


def check_sample(sample: Path, work: Path, timeout: int) -> dict:
    result = {"sample": sample.name, "ok": False, "stage": None, "errors": [], "seconds": 0.0}
    start = time.time()
    src = sample / "src"

    flags = scan_dir(src)
    if flags:
        result.update(stage="restrictions", errors=[f"{path}: {', '.join(f)}" for path, f in flags.items()])
        return result

    bad_json = invalid_json_files(src)
    if bad_json:
        result.update(stage="json", errors=bad_json)
        return result

    shutil.rmtree(work / "src", ignore_errors=True)
    shutil.copytree(src, work / "src")
    try:
        proc = subprocess.run(
            [GRADLEW, *COMPILE_TASKS, "--console=plain", "--stacktrace"],
            cwd=work, capture_output=True, text=True, timeout=timeout, shell=(os.name == "nt"),
        )
    except subprocess.TimeoutExpired:
        result.update(stage="timeout", errors=[f"compile took longer than {timeout}s"])
        return result

    result["seconds"] = round(time.time() - start, 1)
    if proc.returncode == 0:
        result.update(ok=True, stage="compiled")
    else:
        result.update(stage="compile", errors=extract_errors(proc.stdout + "\n" + proc.stderr))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="folder whose subfolders are samples")
    parser.add_argument("--template", default="1.21.11")
    parser.add_argument("--out", default="results/compile.jsonl")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    samples = sorted(p for p in (ROOT / args.samples).iterdir() if (p / "src").is_dir())
    if not samples:
        sys.exit(f"no samples with a src/ folder in {args.samples}")

    work = prepare_work_project(args.template)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    passed = 0
    with out.open("w", encoding="utf-8") as f:
        for i, sample in enumerate(samples, 1):
            result = check_sample(sample, work, args.timeout)
            passed += result["ok"]
            f.write(json.dumps(result) + "\n")
            f.flush()
            status = "PASS" if result["ok"] else f"FAIL ({result['stage']})"
            print(f"[{i}/{len(samples)}] {sample.name}: {status} {result['seconds']}s", flush=True)
            for err in result["errors"][:3]:
                print("    " + err.replace("\n", "\n    "), flush=True)

    print(f"\n{passed}/{len(samples)} samples passed. Results: {out}")


if __name__ == "__main__":
    main()
