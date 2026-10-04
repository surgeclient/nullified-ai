"""Collect real, working, permissively-licensed Fabric 1.21.x mods from GitHub as training data.

Models don't know Minecraft 1.21.11 well, so distilling from them yields almost nothing. Real published mods
ARE correct. This finds small/medium Fabric mods for 1.21.x that use Mojang mappings (our template's mappings),
checks the license, optionally verifies they build, and writes records in the SAME format the trainer reads
(tools/build_sft.py), so they plug straight into training.

    HF_TOKEN=... GITHUB_TOKEN=... python tools/collect_repos.py --run repos1 --max-repos 60 --build
Output: <you>/nullified-ai-data/raw/<run>.jsonl  (records: {id, request, files, ok, stage, source, license})
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = "1.21.11"
PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "unlicense", "mpl-2.0", "cc0-1.0", "isc", "zlib",
              "0bsd", "wtfpl"}
# Java/resource files we take; skip gradle wrapper, tests, datagen clutter is fine to keep but cap total size.
SAFE_REL = re.compile(r"^src/(main|client)/(java|resources)/[\w./\-]+\.(java|json|mcmeta)$")


def gh_api(path: str, token: str, params: dict | None = None):
    url = "https://api.github.com" + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}",
                                               "Accept": "application/vnd.github+json",
                                               "X-GitHub-Api-Version": "2022-11-28"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 403 and attempt < 4:  # secondary rate limit
                time.sleep(15 * (attempt + 1))
                continue
            raise


def find_repos(token: str, queries: list[str], per_query: int, pages: int = 1) -> list[dict]:
    seen, out = set(), []
    for q in queries:
        for order in ("desc", "asc"):  # stars desc AND asc widens past the same popular top-N each run
            for page in range(1, pages + 1):
                try:
                    res = gh_api("/search/repositories", token,
                                 {"q": q, "sort": "stars", "order": order,
                                  "per_page": min(per_query, 100), "page": page})
                except Exception as e:
                    print(f"  search failed for {q!r} p{page}: {str(e)[:120]}", flush=True)
                    break
                items = res.get("items", [])
                for item in items:
                    if item["full_name"] in seen or item.get("fork") or item.get("archived"):
                        continue
                    seen.add(item["full_name"])
                    lic = ((item.get("license") or {}).get("spdx_id") or "").lower()
                    out.append({"full_name": item["full_name"], "clone_url": item["clone_url"],
                                "default_branch": item["default_branch"], "license": lic,
                                "description": item.get("description") or "",
                                "stars": item.get("stargazers_count", 0)})
                if len(items) < min(per_query, 100):
                    break  # no more pages for this query/order
    return out


def find_repos_by_code(token: str, code_queries: list[str], per_query: int, pages: int = 1) -> list[dict]:
    """Find repos directly by a marker in their files (e.g. officialMojangMappings in build.gradle).

    This targets the exact mods we keep instead of filtering them out of a generic search, so it finds
    far more Mojang-mapped mods than repository search alone. Code search only returns the repo stub, so
    we resolve each via /repos/{full} to get license/default_branch/stars.
    """
    full_names = []
    for q in code_queries:
        for page in range(1, pages + 1):
            try:
                res = gh_api("/search/code", token, {"q": q, "per_page": min(per_query, 100), "page": page})
            except Exception as e:
                print(f"  code search failed for {q!r} p{page}: {str(e)[:120]}", flush=True)
                break
            items = res.get("items", [])
            for item in items:
                repo = item.get("repository") or {}
                fn = repo.get("full_name")
                if fn and not repo.get("fork") and fn not in full_names:
                    full_names.append(fn)
            time.sleep(2)  # code search has a tight secondary rate limit
            if len(items) < min(per_query, 100):
                break
    out = []
    for fn in full_names:
        try:
            info = gh_api(f"/repos/{fn}", token)
        except Exception as e:
            print(f"  resolve failed for {fn}: {str(e)[:120]}", flush=True)
            continue
        if info.get("archived") or info.get("fork"):
            continue
        out.append({"full_name": fn, "clone_url": info["clone_url"], "default_branch": info["default_branch"],
                    "license": ((info.get("license") or {}).get("spdx_id") or "").lower(),
                    "description": info.get("description") or "", "stars": info.get("stargazers_count", 0)})
    return out


def is_mojang_fabric_121(repo_dir: Path) -> tuple[bool, str]:
    """Keep only Fabric mods for 1.21.x using Mojang mappings (what our template uses)."""
    mod_jsons = list(repo_dir.rglob("fabric.mod.json"))
    if not mod_jsons:
        return False, "not a Fabric mod (no fabric.mod.json)"
    gradle = ""  # multi-module: mappings may live in a subproject's build.gradle, so scan recursively
    for f in repo_dir.rglob("*"):
        if f.is_file() and (f.name.endswith((".gradle", ".gradle.kts")) or f.name == "gradle.properties"):
            gradle += f.read_text(encoding="utf-8", errors="ignore") + "\n"
    if re.search(r'net\.fabricmc:yarn', gradle) and "officialMojangMappings" not in gradle:
        return False, "uses Yarn mappings (wrong API names for our template)"
    if "officialMojangMappings" not in gradle and "mojmap" not in gradle.lower():
        return False, "mappings not clearly Mojang"
    text = " ".join(f.read_text(encoding="utf-8", errors="ignore") for f in mod_jsons)
    if "1.21" not in text and "1.21" not in gradle:
        return False, "not targeting 1.21.x"
    return True, "ok"


def collect_files(repo_dir: Path) -> dict[str, str]:
    """The mod's source + resources, keyed the way our template expects. Skips oversized mods."""
    # Fabric mods already use src/main/java etc. Find the module that holds fabric.mod.json.
    mod_json = min(repo_dir.rglob("src/main/resources/fabric.mod.json"), key=lambda p: len(p.parts), default=None)
    if mod_json is None:
        return {}
    module = mod_json.parents[3]  # .../<module>/src/main/resources/fabric.mod.json -> <module>
    files, total = {}, 0
    for f in module.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(module).as_posix()
        if not SAFE_REL.match(rel):
            continue
        try:
            body = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        total += len(body)
        files[rel] = body
        if len(files) > 40 or total > 60000:  # too big to be a clean training example
            return {}
    return files


def request_for(repo: dict, files: dict) -> str:
    desc = repo["description"].strip()
    try:
        mod = json.loads(files["src/main/resources/fabric.mod.json"])
        name, mdesc = mod.get("name", ""), (mod.get("description") or "").strip()
    except (KeyError, json.JSONDecodeError):
        name, mdesc = "", ""
    best = max([desc, mdesc], key=len) or name or repo["full_name"].split("/")[-1]
    return f"Build a Fabric mod for Minecraft {VERSION}: {best}"


def build_ok(repo_dir: Path, timeout: int) -> bool:
    gradlew = repo_dir / "gradlew"
    if not gradlew.exists():
        return False
    gradlew.chmod(0o755)
    try:
        p = subprocess.run(["./gradlew", "build", "-x", "test", "--no-daemon", "--console=plain"],
                           cwd=repo_dir, capture_output=True, timeout=timeout)
        return p.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--max-repos", type=int, default=60)
    p.add_argument("--per-query", type=int, default=30)
    p.add_argument("--pages", type=int, default=2, help="search result pages to walk per query (widens the pool)")
    p.add_argument("--build", action="store_true", help="verify each mod actually builds (slow but real)")
    p.add_argument("--build-timeout", type=int, default=900)
    p.add_argument("--repos", default="", help="comma-separated owner/repo to use instead of searching")
    p.add_argument("--data-repo", default="")
    args = p.parse_args()

    gh_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    hf_token = os.environ.get("HF_TOKEN")
    if not gh_token:
        sys.exit("GITHUB_TOKEN required (to search and clone).")

    if args.repos:
        repos = []
        for full in [r.strip() for r in args.repos.split(",") if r.strip()]:
            info = gh_api(f"/repos/{full}", gh_token)
            repos.append({"full_name": full, "clone_url": info["clone_url"], "default_branch": info["default_branch"],
                          "license": ((info.get("license") or {}).get("spdx_id") or "").lower(),
                          "description": info.get("description") or "", "stars": info.get("stargazers_count", 0)})
    else:
        # Code search targets the exact mods we keep (Mojang mappings), so it yields far more than generic
        # repo search, which mostly surfaces Yarn example/tutorial mods we reject. Run it first, over several
        # version strings and pages so we see beyond the same popular top-N each run.
        code_queries = [f'officialMojangMappings {v} filename:build.gradle' for v in ("1.21", "1.21.1", "1.21.11")]
        code_queries += ['officialMojangMappings fabric filename:build.gradle.kts',
                         'mappings loom.officialMojangMappings() filename:build.gradle']
        repos = find_repos_by_code(gh_token, code_queries, args.per_query, pages=args.pages)
        code_names = {r["full_name"] for r in repos}
        print(f"{len(repos)} repos from code search (Mojang-mapped)", flush=True)
        queries = ['fabric minecraft mod 1.21 language:Java', 'fabric-mod 1.21 language:Java',
                   'minecraft fabric 1.21.1 language:Java', 'fabric mod 1.21.11 language:Java',
                   'fabric example mod 1.21 language:Java', 'fabric mod tutorial 1.21 language:Java']
        repos += [r for r in find_repos(gh_token, queries, args.per_query, pages=args.pages)
                  if r["full_name"] not in code_names]
    print(f"{len(repos)} candidate repos", flush=True)

    records, work = [], Path(tempfile.mkdtemp(prefix="nai-repos-"))
    for repo in repos:
        if len(records) >= args.max_repos:
            break
        full = repo["full_name"]
        if repo["license"] not in PERMISSIVE:
            print(f"skip {full}: license '{repo['license'] or 'none'}' not permissive", flush=True)
            continue
        d = work / full.replace("/", "__")
        try:
            subprocess.run(["git", "clone", "--depth", "1", "--branch", repo["default_branch"],
                            repo["clone_url"], str(d)], capture_output=True, timeout=300, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"skip {full}: clone failed", flush=True)
            continue
        ok_map, why = is_mojang_fabric_121(d)
        if not ok_map:
            print(f"skip {full}: {why}", flush=True)
            shutil.rmtree(d, ignore_errors=True)
            continue
        files = collect_files(d)
        if not files or "src/main/resources/fabric.mod.json" not in files:
            print(f"skip {full}: no usable src (too big or unusual layout)", flush=True)
            shutil.rmtree(d, ignore_errors=True)
            continue
        verified = True
        if args.build:
            verified = build_ok(d, args.build_timeout)
            print(f"{'built' if verified else 'build FAILED'} {full} ({len(files)} files)", flush=True)
        if not verified:
            shutil.rmtree(d, ignore_errors=True)
            continue
        records.append({"id": f"{args.run}-{len(records):04d}", "run": args.run, "version": VERSION,
                        "source": f"github:{full}", "license": repo["license"], "stars": repo["stars"],
                        "request": request_for(repo, files), "plan": "", "files": files,
                        "refused": None, "ok": True, "stage": "passed" if args.build else "unverified",
                        "errors": [], "attempts": []})
        print(f"[{len(records)}] kept {full} ({len(files)} files, {repo['license']})", flush=True)
        shutil.rmtree(d, ignore_errors=True)

    out = ROOT / "output" / "data"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.run}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    summary = {"run": args.run, "kept": len(records), "built": args.build,
               "sources": [r["source"] for r in records]}
    (out / f"{args.run}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("run", "kept", "built")}), flush=True)

    if hf_token and records:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        repo_id = args.data_repo or f"{api.whoami()['name']}/nullified-ai-data"
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(path), path_in_repo=f"raw/{args.run}.jsonl",
                        repo_id=repo_id, repo_type="dataset")
        print(f"uploaded -> {repo_id}/raw/{args.run}.jsonl", flush=True)
    elif not hf_token:
        print("WARNING: no HF_TOKEN; saved locally only", flush=True)


if __name__ == "__main__":
    main()
