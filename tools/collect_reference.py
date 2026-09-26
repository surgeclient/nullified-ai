"""Build the reference corpus the teachers and the docs lookup use.

Sources (cloned by the collect-reference workflow):
  - fabric-docs: version-specific docs + the reference mod for each version
  - fabric (Fabric API): public API source for the target version
  - Minecraft: public class/method signatures from the Mojang-mapped game jar (names only, no code)

Output: one JSONL file per source in --out. Each line:
  {"id", "version", "source", "license", "kind", "path", "title", "text"}

Usage:
    python tools/collect_reference.py --docs ext/fabric-docs --api ext/fabric --version 1.21.11 --out reference
"""
import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

MAX_CHARS = 8000
DOCS_LICENSE = "CC-BY-NC-SA-4.0"
API_LICENSE = "Apache-2.0"


def record(version, source, license_, kind, path, title, text):
    uid = hashlib.sha1(f"{source}:{path}:{title}".encode()).hexdigest()[:16]
    return {"id": uid, "version": version, "source": source, "license": license_,
            "kind": kind, "path": path, "title": title, "text": text}


def split_long(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Split on blank lines so chunks stay under limit."""
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for block in re.split(r"\n\s*\n", text):
        if current and len(current) + len(block) > limit:
            chunks.append(current)
            current = ""
        current += block + "\n\n"
    if current.strip():
        chunks.append(current)
    return chunks


def collect_docs(docs_root: Path, version: str, docs_dir: str, ref_dir: str):
    """Markdown docs split by ## heading, plus every source/resource file of the reference mod."""
    for md in sorted((docs_root / docs_dir).rglob("*.md")):
        rel = md.relative_to(docs_root).as_posix()
        text = md.read_text(encoding="utf-8", errors="ignore")
        page_title = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), md.stem)
        for section in re.split(r"(?m)^(?=## )", text):
            if not section.strip():
                continue
            heading = section.splitlines()[0].lstrip("# ").strip()
            title = page_title if heading == page_title else f"{page_title} > {heading}"
            for i, chunk in enumerate(split_long(section)):
                yield record(version, "fabric-docs", DOCS_LICENSE, "doc", rel, f"{title} #{i}", chunk.strip())

    ref_root = docs_root / ref_dir
    for file in sorted(ref_root.rglob("*")):
        if file.is_file() and file.suffix in {".java", ".json", ".classtweaker"} and "/build/" not in file.as_posix():
            rel = file.relative_to(docs_root).as_posix()
            text = file.read_text(encoding="utf-8", errors="ignore")
            for i, chunk in enumerate(split_long(text)):
                yield record(version, "fabric-docs-reference", DOCS_LICENSE, "example", rel, f"{file.name} #{i}", chunk)


def collect_fabric_api(api_root: Path, version: str):
    """Public Fabric API classes only (packages under .../api/), with javadoc."""
    for java in sorted(api_root.rglob("*.java")):
        rel = java.relative_to(api_root).as_posix()
        if "/net/fabricmc/fabric/api/" not in rel or "/test" in rel or "testmod" in rel:
            continue
        text = java.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r"^/\*.*?Licensed under the Apache License.*?\*/\s*", "", text, flags=re.S)  # drop header
        module = rel.split("/")[0]
        for i, chunk in enumerate(split_long(text)):
            yield record(version, "fabric-api", API_LICENSE, "api", rel, f"{module}: {java.stem} #{i}", chunk)


def find_minecraft_jars(search_dirs: list[Path]) -> list[Path]:
    """Loom's Mojang-mapped game jars: the ones containing Minecraft classes."""
    jars = []
    for jar in (j for d in search_dirs if d.exists() for j in d.rglob("*.jar")):
        name = jar.name.lower()
        if "minecraft" not in name or "sources" in name or jar.stat().st_size < 5_000_000:
            continue
        try:
            with zipfile.ZipFile(jar) as z:
                if "net/minecraft/resources/Identifier.class" in z.namelist() or \
                        "net/minecraft/client/Minecraft.class" in z.namelist():
                    jars.append(jar)
        except zipfile.BadZipFile:
            pass
    return jars


def collect_minecraft_signatures(jars: list[Path], version: str, batch: int = 300):
    """Public API surface via javap: class, field and method signatures only."""
    seen = set()
    for jar in jars:
        with zipfile.ZipFile(jar) as z:
            classes = sorted(
                n[:-6].replace("/", ".") for n in z.namelist()
                if n.startswith(("net/minecraft/", "com/mojang/")) and n.endswith(".class")
                and not re.search(r"\$\d", n)
            )
        classes = [c for c in classes if c not in seen]
        seen.update(classes)
        for start in range(0, len(classes), batch):
            out = subprocess.run(["javap", "-public", "-cp", str(jar), *classes[start:start + batch]],
                                 capture_output=True, text=True).stdout
            for block in re.split(r"(?m)^Compiled from .*\n", out):
                block = block.strip()
                if not block:
                    continue
                header = block.splitlines()[0]
                m = re.search(r"(?:class|interface|enum|record)\s+([\w.$]+)", header)
                if m:
                    yield record(version, "minecraft", "Mojang (signatures only)", "signature",
                                 m.group(1), m.group(1), block)


def write(path: Path, records) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            if r["text"].strip():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True, help="fabric-docs checkout")
    parser.add_argument("--api", required=True, help="fabric (Fabric API) checkout for --version")
    parser.add_argument("--version", default="1.21.11")
    parser.add_argument("--jar-search", nargs="+", default=[str(Path.home() / ".gradle")],
                        help="folders to search for Loom's mapped Minecraft jars")
    parser.add_argument("--out", default="reference")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = Path(args.docs)
    v = args.version

    counts = {
        f"docs-{v}": write(out / f"docs-{v}.jsonl",
                           collect_docs(docs, v, f"versions/{v}/develop", f"reference/{v}")),
        "docs-latest": write(out / "docs-latest.jsonl", collect_docs(docs, "latest", "develop", "reference/latest")),
        f"fabric-api-{v}": write(out / f"fabric-api-{v}.jsonl", collect_fabric_api(Path(args.api), v)),
    }
    jars = find_minecraft_jars([Path(d) for d in args.jar_search])
    print("minecraft jars:", *[j.name for j in jars], sep="\n  ")
    counts[f"minecraft-{v}"] = write(out / f"minecraft-{v}.jsonl", collect_minecraft_signatures(jars, v))

    (out / "manifest.json").write_text(json.dumps({"version": v, "counts": counts}, indent=2))
    for name, n in counts.items():
        print(f"{name}: {n} records")


if __name__ == "__main__":
    main()
