"""Run Nullified AI without Modal: the model runs free on your own GPU, the Minecraft checks run free on GitHub Actions.

Nothing Minecraft-related is stored on your PC: compiling, booting a real 1.21.11 server and building the
.jar/.zip all happen in the check-mod workflow, and only the finished .jar + project .zip are downloaded.
The only thing on your disk is the model itself (qwen3:8b is ~5 GB and fits an 8 GB GPU).

Setup: see local/README.md. Then:
    python local/nullified_local.py build --request "a ruby sword that sets mobs on fire"
    python local/nullified_local.py evaluate --limit 5

Same pipeline as modal_jobs/nullified.py: restrictions gate -> plan + write -> compile -> boot a server ->
check assets -> the model fixes whatever failed -> package .jar + .zip.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from prompts import (FIX_PROMPT, SOLVE_PROMPT, STAGE_EXPLAINED, format_files, looks_looped,  # noqa: E402
                     parse_answer, parse_review, review_prompt, runtime_hints, system_prompt)

VERSION = "1.21.11"
THINK = re.compile(r"<think>.*?</think>", re.S)


# ---------------------------------------------------------------- the model, on your GPU

class LocalModel:
    """Ollama's own API by default (lets us set the context size and turn thinking off per request).
    An --api-base ending in /v1 uses the OpenAI-style API instead (LM Studio, llama.cpp server, vLLM)."""

    def __init__(self, api_base: str, model: str, context: int, parallel: int):
        self.api_base, self.model, self.context, self.parallel = api_base.rstrip("/"), model, context, parallel
        self.openai = self.api_base.endswith("/v1")
        self.think_flag = True  # dropped automatically if the server/model rejects it
        self.system = system_prompt(VERSION)

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.api_base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
        with urllib.request.urlopen(req, timeout=3 * 3600) as resp:
            return json.loads(resp.read())

    def _one(self, prompt: str, temperature: float, max_tokens: int, repetition_penalty: float) -> dict:
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": prompt}]
        max_tokens = min(max_tokens, self.context // 2)
        try:
            if self.openai:
                choice = self._post("/chat/completions", {
                    "model": self.model, "messages": messages, "temperature": temperature, "top_p": 0.95,
                    "max_tokens": max_tokens, "repetition_penalty": repetition_penalty,
                    "chat_template_kwargs": {"enable_thinking": False}})["choices"][0]
                text, finish = choice["message"].get("content") or "", choice.get("finish_reason")
            else:
                body = {"model": self.model, "messages": messages, "stream": False,
                        "options": {"num_ctx": self.context, "num_predict": max_tokens, "temperature": temperature,
                                    "top_p": 0.95, "repeat_penalty": repetition_penalty}}
                if self.think_flag:
                    body["think"] = False
                try:
                    out = self._post("/api/chat", body)
                except urllib.error.HTTPError as e:
                    if e.code != 400 or "think" not in body:
                        raise
                    self.think_flag = False  # model without a thinking switch
                    body.pop("think")
                    out = self._post("/api/chat", body)
                text, finish = out["message"].get("content") or "", out.get("done_reason")
        except urllib.error.HTTPError as e:
            raise SystemExit(f"Model server error {e.code}: {e.read().decode(errors='ignore')[:500]}\n"
                             f"Is the model downloaded? (ollama pull {self.model})")
        except urllib.error.URLError as e:
            raise SystemExit(f"Could not reach the model server at {self.api_base}: {e.reason}\n"
                             "Start Ollama first (open the Ollama app, or run `ollama serve`).")
        return {"text": THINK.sub("", text).strip(), "finish": finish}

    def chat(self, prompts: list[str], temperature: float, max_tokens: int = 14000,
             repetition_penalty: float = 1.05) -> list[dict]:
        with ThreadPoolExecutor(self.parallel) as pool:
            return list(pool.map(lambda p: self._one(p, temperature, max_tokens, repetition_penalty), prompts))


def stuck(out: dict) -> bool:
    return out["finish"] == "length" or looks_looped(out["text"])


# ---------------------------------------------------------------- reference docs (optional, but much better)

class _NoReference:
    def context_for(self, *_args, **_kwargs) -> str:
        return ""

    def hints_for_errors(self, *_args, **_kwargs) -> str:
        return ""


def load_reference(folder: str):
    """1.21.11 docs + class signatures: --reference-dir, else your private HF dataset (needs HF_TOKEN)."""
    from retrieval import ReferenceIndex
    from symbols import SymbolIndex
    path = Path(folder) if folder else None
    if path is None and os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError:
            sys.exit("HF_TOKEN is set, so reference docs will be downloaded: pip install huggingface_hub")
        token = os.environ["HF_TOKEN"]
        user = HfApi(token=token).whoami()["name"]
        path = Path(snapshot_download(f"{user}/nullified-ai-reference", repo_type="dataset", token=token,
                                      allow_patterns=[f"{VERSION}/*"], local_dir=str(ROOT / ".work" / "reference"))) / VERSION
    if path is None or not (path / f"minecraft-{VERSION}.jsonl").exists():
        print("Note: no reference docs (set HF_TOKEN or pass --reference-dir). The model will work from memory, "
              "which fails more often.")
        return _NoReference(), _NoReference()
    return ReferenceIndex.load(path, VERSION), SymbolIndex.load(path / f"minecraft-{VERSION}.jsonl")


# ---------------------------------------------------------------- checks: GitHub Actions (default) or this PC

def github_repo() -> str:
    if os.environ.get("NULLIFIED_REPO"):
        return os.environ["NULLIFIED_REPO"]
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True).stdout
    except OSError:
        url = ""
    m = re.search(r"github\.com[:/]([^/]+/[^/.\s]+)", url)
    return m.group(1) if m else "surgeclient/nullified-ai"


class Checker:
    def __init__(self, where: str, ref: str):
        self.local = where == "local"
        if self.local:
            import mod_checks  # needs Java 21 + ~3 GB for Gradle and Minecraft
            self.mod_checks = mod_checks
        else:
            from github_checks import GitHubChecks
            token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
            if not token:
                sys.exit("Set GITHUB_TOKEN to a GitHub token with 'Actions: Read and write' + 'Contents: Read' "
                         "on this repo (see local/README.md), or use --checks local.")
            self.gh = GitHubChecks(github_repo(), token, ref)

    def check(self, items: list[dict]) -> list[dict]:
        if self.local:
            return [self.mod_checks.check(i["id"], i["files"]) for i in items]
        results = []
        for group in self.gh.split(items):
            results += self.gh.run({"items": group})["checks"]
        return results

    def package(self, files: dict[str, str], out: Path) -> dict:
        if self.local:
            pkg = self.mod_checks.package(files, out / "tmp")
            folder = out / pkg["mod_id"]
            folder.mkdir(parents=True, exist_ok=True)
            for name in (pkg["jar"], pkg["zip"]):
                (out / "tmp" / name).replace(folder / name)
            return {**pkg, "folder": str(folder.resolve())}
        return self.gh.run({"items": [{"id": "package", "files": files}], "package": True}, save_to=out)["package"]


# ---------------------------------------------------------------- the plan -> write -> check -> fix loop

def solve(llm: LocalModel, checker: Checker, index, symbols, requests: list[str], fix_rounds: int) -> list[dict]:
    print("Checking requests against restrictions.txt...", flush=True)
    blocked = [parse_review(o["text"]) for o in llm.chat([review_prompt(r) for r in requests], 0.0, max_tokens=200)]
    records = [{"id": f"req{i}", "request": req, "plan": "", "files": {}, "refused": rule, "ok": False,
                "stage": "refused" if rule else None, "errors": [], "stage_history": []}
               for i, (req, rule) in enumerate(zip(requests, blocked))]

    allowed = [r for r in records if not r["refused"]]
    print(f"Writing {len(allowed)} mod(s) on your GPU...", flush=True)
    budget = min(36000, llm.context)  # chars of reference docs; keeps prompt + answer inside the context window
    prompts = [SOLVE_PROMPT.format(version=VERSION, request=r["request"],
                                   reference=index.context_for(r["request"], budget_chars=budget))
               for r in allowed]
    outs = llm.chat(prompts, 0.4)
    retry = [i for i, o in enumerate(outs) if stuck(o)]
    for i, o in zip(retry, llm.chat([prompts[i] for i in retry], 0.8, repetition_penalty=1.15)):
        if not stuck(o) or len(parse_answer(o["text"])["files"]) > len(parse_answer(outs[i]["text"])["files"]):
            outs[i] = o
    for r, out in zip(allowed, outs):
        parsed = parse_answer(out["text"])
        r.update(plan=parsed["plan"], files=parsed["files"], refused=parsed["refused"],
                 stage="refused" if parsed["refused"] else ("no-files" if not parsed["files"] else None),
                 first_answer=out["text"], first_finish=out["finish"])

    for rnd in range(fix_rounds + 1):
        todo = [r for r in records if r["files"] and not r["ok"]]
        if todo:
            print(f"Round {rnd}: checking {len(todo)} mod(s)...", flush=True)
            by_id = {r["id"]: r for r in records}
            for res in checker.check([{"id": r["id"], "files": r["files"]} for r in todo]):
                by_id[res["id"]].update(ok=res["ok"], stage=res["stage"], errors=res["errors"])
        for r in records:
            r["stage_history"].append(r["stage"])
        failing = [r for r in records if r["files"] and not r["ok"] and r["stage"] in STAGE_EXPLAINED]
        print(f"Round {rnd}: {sum(r['ok'] for r in records)}/{len(records)} pass", flush=True)
        if rnd == fix_rounds or not failing:
            break
        print(f"Round {rnd}: fixing {len(failing)} mod(s) on your GPU...", flush=True)
        prompts = [FIX_PROMPT.format(version=VERSION, stage=STAGE_EXPLAINED[r["stage"]], request=r["request"],
                                     files=format_files(r["files"]),
                                     errors="\n\n".join(e[:600] for e in r["errors"][:15]),
                                     hints=symbols.hints_for_errors(r["errors"], r["files"])[:min(20000, budget)]
                                     + runtime_hints(r["errors"]))
                   for r in failing]
        for r, out in zip(failing, llm.chat(prompts, 0.2)):
            if not stuck(out):  # a looping fix would overwrite good files with garbage
                r["files"] = {**r["files"], **parse_answer(out["text"])["files"]}
    return records


# ---------------------------------------------------------------- commands

def cmd_build(args, llm, checker, index, symbols):
    started = time.time()
    rec = solve(llm, checker, index, symbols, [args.request], args.fix_rounds)[0]
    if rec["refused"]:
        print(f"Refused: {rec['refused']}")
        return
    print(f"checks per round: {' -> '.join(str(s) for s in rec['stage_history'])}")
    if rec["stage"] == "restrictions":
        print("Blocked: the generated code matched a pattern from restrictions.txt, so no jar was built:")
        for e in rec["errors"][:5]:
            print("  " + e)
        return
    if not rec["ok"]:
        print("Could not produce a mod that passes every check. Last problems:")
        for e in rec["errors"][:8]:
            lines = e.splitlines() or [""]
            print("  " + lines[0][:200])
            for cause in [l.strip() for l in lines[1:] if l.strip().startswith("Caused by")][-1:]:
                print("    " + cause[:200])  # the root cause of a crash
        return
    print("Packaging the .jar and project .zip...", flush=True)
    pkg = checker.package(rec["files"], ROOT / "output")
    print(f"PLAN:\n{rec['plan']}\n\nDone in {(time.time() - started) / 60:.1f} min -> {pkg['folder']}")
    if pkg["placeholder_textures"]:
        print(f"{len(pkg['placeholder_textures'])} placeholder textures were added (replace them with real art):")
        for t in pkg["placeholder_textures"]:
            print("  " + t)


def cmd_evaluate(args, llm, checker, index, symbols):
    requests = [e["request"] for e in json.loads((ROOT / "tools" / "eval_requests.json").read_text())]
    requests = requests[:args.limit] if args.limit else requests
    recs = solve(llm, checker, index, symbols, requests, args.fix_rounds)
    rounds = max(len(r["stage_history"]) for r in recs)
    per_round = [sum(1 for r in recs if len(r["stage_history"]) > k and r["stage_history"][k] == "passed"
                     or (len(r["stage_history"]) <= k and r["ok"])) for k in range(rounds)]
    # Every eval request is allowed, so any refusal here is a wrong refusal.
    report = {"model": args.model, "first_try": per_round[0], "after_fixes": per_round[-1], "per_round": per_round,
              "wrongly_refused": sum(bool(r["refused"]) for r in recs), "of": len(recs),
              "final_stages": {s: sum(r["stage"] == s for r in recs) for s in {r["stage"] for r in recs}},
              "cut_off": sum(r.get("first_finish") == "length" for r in recs)}
    print(json.dumps(report, indent=2))
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    name = re.sub(r"[^\w.-]", "_", args.model)
    with open(out / f"eval_local_{name}.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            print(json.dumps(r, ensure_ascii=False), file=f)
    (out / f"eval_local_{name}.json").write_text(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("build", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--model", default=os.environ.get("NULLIFIED_MODEL", "qwen3:8b"),
                       help="model name as your server knows it (env NULLIFIED_MODEL). Default fits an 8 GB GPU")
        s.add_argument("--api-base", default=os.environ.get("NULLIFIED_API_BASE", "http://localhost:11434"),
                       help="Ollama URL (default), or an OpenAI-style URL ending in /v1 (LM Studio, llama.cpp)")
        s.add_argument("--context", type=int, default=int(os.environ.get("NULLIFIED_CONTEXT", 16384)),
                       help="context window in tokens; 16384 fits 8 GB of VRAM next to an 8B model")
        s.add_argument("--parallel", type=int, default=1, help="requests sent to the model at once")
        s.add_argument("--checks", choices=["github", "local"], default="github",
                       help="github (default): compile/test on GitHub Actions, nothing stored locally. "
                            "local: needs Java 21 + ~3 GB")
        s.add_argument("--ref", default="", help="branch the check-mod workflow runs on (default: repo default)")
        s.add_argument("--reference-dir", default="", help="folder with the 1.21.11 reference .jsonl files")
        s.add_argument("--fix-rounds", type=int, default=4 if name == "build" else 3)
        if name == "build":
            s.add_argument("--request", required=True)
        else:
            s.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    llm = LocalModel(args.api_base, args.model, args.context, args.parallel)
    checker = Checker(args.checks, args.ref)
    index, symbols = load_reference(args.reference_dir)
    {"build": cmd_build, "evaluate": cmd_evaluate}[args.command](args, llm, checker, index, symbols)


if __name__ == "__main__":
    main()
