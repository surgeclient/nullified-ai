"""Turn one mod sample into a full Gradle project, build the .jar, and zip the project.

Usage:
    python tools/build_mod.py --sample samples/smoke/ruby_item --template 1.21.11 --out results/build
Outputs <modid>.jar and <modid>-project.zip in --out.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from restrictions import scan_dir  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GRADLEW = "gradlew.bat" if os.name == "nt" else "./gradlew"


def read_mod_id(src: Path) -> str:
    mod_json = src / "main" / "resources" / "fabric.mod.json"
    mod_id = json.loads(mod_json.read_text(encoding="utf-8"))["id"]
    if not re.fullmatch(r"[a-z][a-z0-9_-]{1,63}", mod_id):
        sys.exit(f"invalid mod id: {mod_id!r}")
    return mod_id


def assemble_project(sample: Path, template: str, dest: Path, mod_id: str) -> None:
    """Template build files + the sample's src/, renamed to the mod's id."""
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(ROOT / "templates" / template, dest, ignore=shutil.ignore_patterns("src"))
    shutil.copytree(sample / "src", dest / "src")

    settings = dest / "settings.gradle"
    settings.write_text(re.sub(r"rootProject\.name = '.*'", f"rootProject.name = '{mod_id}'",
                               settings.read_text(encoding="utf-8")), encoding="utf-8")
    build = dest / "build.gradle"
    build.write_text(build.read_text(encoding="utf-8").replace('"modid" {', f'"{mod_id}" {{'), encoding="utf-8")
    if os.name != "nt":
        (dest / "gradlew").chmod(0o755)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--template", default="1.21.11")
    parser.add_argument("--out", default="results/build")
    args = parser.parse_args()

    sample = ROOT / args.sample
    src = sample / "src"
    flags = scan_dir(src)
    if flags:
        sys.exit(f"blocked by restrictions: {flags}")

    mod_id = read_mod_id(src)
    project = ROOT / ".work" / "build" / mod_id
    assemble_project(sample, args.template, project, mod_id)

    proc = subprocess.run([GRADLEW, "build", "--console=plain"], cwd=project, shell=(os.name == "nt"))
    if proc.returncode != 0:
        sys.exit("gradle build failed")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    jars = [j for j in (project / "build" / "libs").glob("*.jar") if not j.name.endswith("-sources.jar")]
    for jar in jars:
        shutil.copy2(jar, out / jar.name)

    # Zip the clean project (no build output or Gradle caches).
    clean = ROOT / ".work" / "zip" / mod_id
    shutil.rmtree(clean, ignore_errors=True)
    shutil.copytree(project, clean, ignore=shutil.ignore_patterns("build", ".gradle", "run"))
    shutil.make_archive(str(out / f"{mod_id}-project"), "zip", clean)

    print(f"built: {', '.join(j.name for j in jars)} and {mod_id}-project.zip in {out}")


if __name__ == "__main__":
    main()
