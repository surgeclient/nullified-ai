"""Run Nullified AI without Modal. Nothing Minecraft-related is stored on your PC.

The model: a big model on a free online API (OpenRouter, Groq, Cerebras - no disk space), or a model on your
own GPU through Ollama. The checks: compiling, booting a real 1.21.11 server and building the .jar/.zip run in
the check-mod workflow on GitHub Actions; only the finished .jar + project .zip are downloaded.

Only restrictions.txt decides what gets refused. If the online AI refuses something on its own rules (anything
restrictions.txt doesn't cover), the request is handed to your local fallback model (qwen3:8b on your GPU).

Setup: see local/README.md. Then:
    python local/nullified_local.py models                      # free models your provider offers
    python local/nullified_local.py build --model <id> --request "a ruby sword that sets mobs on fire"
    python local/nullified_local.py evaluate --model <id> --limit 5

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
from restrictions import load_restrictions  # noqa: E402
from prompts import (FIX_PROMPT, SOLVE_PROMPT, STAGE_EXPLAINED, format_files, looks_looped,  # noqa: E402
                     parse_answer, parse_review, review_prompt, runtime_hints, system_prompt)

VERSION = "1.21.11"
THINK = re.compile(r"<think>.*?</think>", re.S)


# ---------------------------------------------------------------- the model (online API or your GPU)

PROVIDERS = {  # name -> (api base, env var holding the key, where to get a free key)
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "https://openrouter.ai/keys"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "https://console.groq.com/keys"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "https://cloud.cerebras.ai"),
    "ollama": ("http://localhost:11434", "", "https://ollama.com/download"),
    "server": ("http://127.0.0.1:8080/v1", "", ""),  # llama.cpp llama-server (what the GitHub runner uses)
}


class ModelUnavailable(Exception):
    pass


class Model:
    """Ollama's own API for a URL without /v1 (lets us set the context size and turn thinking off per request);
    the OpenAI-style API for a URL ending in /v1 (OpenRouter, Groq, Cerebras, LM Studio, llama.cpp, vLLM)."""

    def __init__(self, api_base: str, model: str, context: int, parallel: int, api_key: str = "", label: str = ""):
        self.api_base, self.model, self.context, self.parallel = api_base.rstrip("/"), model, context, parallel
        self.api_key, self.label = api_key or "local", label or model
        self.openai = self.api_base.endswith("/v1")
        self.think_flag = True    # Ollama: dropped automatically if the model has no thinking switch
        self.extras_ok = True     # OpenAI-style: dropped automatically if the server rejects the extra fields
        self.system = system_prompt(VERSION)

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.api_base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}",
                                              "X-Title": "Nullified AI"})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(req, timeout=3 * 3600) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code not in (429, 502, 503) or attempt == 7:
                    raise
                # Free tiers are rate limited: wait as long as the server asks (or back off), then retry.
                wait = min(float(e.headers.get("Retry-After") or 0) or 10 * 2 ** attempt, 300)
                print(f"    {self.label}: busy/rate limited ({e.code}), retrying in {wait:.0f}s...", flush=True)
                time.sleep(wait)

    def _one(self, prompt: str, temperature: float, max_tokens: int, repetition_penalty: float) -> dict:
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": prompt}]
        max_tokens = min(max_tokens, self.context // 2)
        try:
            if self.openai:
                body = {"model": self.model, "messages": messages, "temperature": temperature, "top_p": 0.95,
                        "max_tokens": max_tokens}
                extras = {"repetition_penalty": repetition_penalty, "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}}
                try:
                    out = self._post("/chat/completions", {**body, **extras} if self.extras_ok else body)
                except urllib.error.HTTPError as e:
                    if e.code != 400 or not self.extras_ok:
                        raise
                    self.extras_ok = False  # strict server: send only the standard fields
                    out = self._post("/chat/completions", body)
                if "choices" not in out:
                    raise ModelUnavailable(f"{self.label}: unexpected reply {str(out)[:300]}")
                choice = out["choices"][0]
                text, finish = (choice.get("message") or {}).get("content") or "", choice.get("finish_reason")
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
                    self.think_flag = False
                    body.pop("think")
                    out = self._post("/api/chat", body)
                text, finish = out["message"].get("content") or "", out.get("done_reason")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="ignore")[:500]
            hint = {401: "the API key is missing or wrong", 402: "this model isn't free - pick one from `models`",
                    404: "unknown model - run `python local/nullified_local.py models`"}.get(e.code, "")
            raise ModelUnavailable(f"{self.label}: error {e.code} {hint}\n{detail}")
        except urllib.error.URLError as e:
            raise ModelUnavailable(f"{self.label}: could not reach {self.api_base} ({e.reason})")
        return {"text": THINK.sub("", text).strip(), "finish": finish}

    def chat(self, prompts: list[str], temperature: float, max_tokens: int = 14000,
             repetition_penalty: float = 1.05) -> list[dict]:
        with ThreadPoolExecutor(self.parallel) as pool:
            return list(pool.map(lambda p: self._one(p, temperature, max_tokens, repetition_penalty), prompts))


class Team:
    """The main model, plus the local fallback that takes over whenever the main one refuses on its own rules.
    Only restrictions.txt may block a request; any other refusal from an online AI is not ours to keep."""

    def __init__(self, main: Model, fallback: Model | None):
        self.main, self.fallback = main, fallback
        self.context = main.context

    def chat(self, prompts: list[str], temperature: float, **kw) -> list[dict]:
        try:
            return self.main.chat(prompts, temperature, **kw)
        except ModelUnavailable as e:
            raise SystemExit(str(e))

    def redo(self, prompts: list[str], temperature: float, why: str, **kw) -> list[dict] | None:
        """Ask the fallback model instead. None if there is no working fallback."""
        if not self.fallback or not prompts:
            return None
        print(f"    {len(prompts)} answer(s) {why} - asking {self.fallback.label} on your GPU instead", flush=True)
        try:
            return self.fallback.chat(prompts, temperature, **kw)
        except ModelUnavailable as e:
            print(f"    fallback unavailable ({e}); keeping {self.main.label}'s answer", flush=True)
            self.fallback = None
            return None


REFUSAL = re.compile(r"^\s*(I'?m sorry|I (can(no|')t|won'?t|am unable|'m unable)|Sorry,|I must decline|"
                     r"As an AI|I'?m not able to)", re.I)
WORD = re.compile(r"[a-z]{4,}")


def self_refused(out: dict) -> bool:
    """The online AI refused by itself: a content filter, or an apology with no files in it."""
    return out["finish"] == "content_filter" or (bool(REFUSAL.match(out["text"])) and "=== FILE:" not in out["text"])


def matching_restriction(reason: str) -> str | None:
    """The restrictions.txt line a gate decision cites, or None if it cites something that isn't in the file."""
    said = set(WORD.findall(reason.lower()))
    best, best_score = None, 0.0
    for rule in load_restrictions():
        words = set(WORD.findall(rule.lower()))
        score = len(said & words) / max(len(words), 1)
        if score > best_score:
            best, best_score = rule, score
    return best if best_score >= 0.5 else None


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

def gate(llm: Team, requests: list[str]) -> list[str | None]:
    """restrictions.txt check. A block only stands if it cites a line of restrictions.txt; a block for any other
    reason (the online AI's own rules) goes to the local fallback model, or is dropped if there is none."""
    print("Checking requests against restrictions.txt...", flush=True)
    prompts = [review_prompt(r) for r in requests]
    outs = llm.chat(prompts, 0.0, max_tokens=200)
    decisions = []
    for out in outs:
        cited = parse_review(out["text"])
        if self_refused(out):
            decisions.append("?")
        elif cited:
            decisions.append(matching_restriction(cited) or "?")
        else:
            decisions.append(None)
    unsure = [i for i, d in enumerate(decisions) if d == "?"]
    redo = llm.redo([prompts[i] for i in unsure], 0.0, "blocked for a reason that is not in restrictions.txt",
                    max_tokens=200)
    for k, i in enumerate(unsure):
        cited = parse_review(redo[k]["text"]) if redo else None
        decisions[i] = matching_restriction(cited) if cited else None
    return decisions


def solve(llm: Team, checker: Checker, index, symbols, requests: list[str], fix_rounds: int) -> list[dict]:
    blocked = gate(llm, requests)
    records = [{"id": f"req{i}", "request": req, "plan": "", "files": {}, "refused": rule, "ok": False,
                "stage": "refused" if rule else None, "errors": [], "stage_history": []}
               for i, (req, rule) in enumerate(zip(requests, blocked))]

    allowed = [r for r in records if not r["refused"]]
    print(f"Writing {len(allowed)} mod(s) with {llm.main.label}...", flush=True)
    budget = min(36000, llm.context)  # chars of reference docs; keeps prompt + answer inside the context window
    prompts = [SOLVE_PROMPT.format(version=VERSION, request=r["request"],
                                   reference=index.context_for(r["request"], budget_chars=budget))
               for r in allowed]
    outs = llm.chat(prompts, 0.4)
    retry = [i for i, o in enumerate(outs) if stuck(o)]
    for i, o in zip(retry, llm.chat([prompts[i] for i in retry], 0.8, repetition_penalty=1.15)):
        if not stuck(o) or len(parse_answer(o["text"])["files"]) > len(parse_answer(outs[i]["text"])["files"]):
            outs[i] = o
    # These requests already passed the restrictions.txt gate, so a refusal now is the online AI's own rule.
    refused = [i for i, o in enumerate(outs) if self_refused(o) or parse_answer(o["text"])["refused"]]
    redo = llm.redo([prompts[i] for i in refused], 0.4, "refused by the online AI's own rules")
    for k, i in enumerate(refused):
        if redo:
            outs[i] = redo[k]
    for r, out in zip(allowed, outs):
        parsed = parse_answer(out["text"])
        if self_refused(out) and not parsed["refused"]:
            parsed["refused"] = f"the online AI refused on its own rules (not restrictions.txt): {out['text'][:200]}"
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
        print(f"Round {rnd}: fixing {len(failing)} mod(s) with {llm.main.label}...", flush=True)
        prompts = [FIX_PROMPT.format(version=VERSION, stage=STAGE_EXPLAINED[r["stage"]], request=r["request"],
                                     files=format_files(r["files"]),
                                     errors="\n\n".join(e[:600] for e in r["errors"][:15]),
                                     hints=symbols.hints_for_errors(r["errors"], r["files"])[:min(20000, budget)]
                                     + runtime_hints(r["errors"]))
                   for r in failing]
        outs = llm.chat(prompts, 0.2)
        refused = [i for i, o in enumerate(outs) if self_refused(o)]
        redo = llm.redo([prompts[i] for i in refused], 0.2, "refused by the online AI's own rules")
        for k, i in enumerate(refused):
            if redo:
                outs[i] = redo[k]
        for r, out in zip(failing, outs):
            if not stuck(out):  # a looping fix would overwrite good files with garbage
                r["files"] = {**r["files"], **parse_answer(out["text"])["files"]}
    return records


# ---------------------------------------------------------------- commands

def cmd_build(args, llm: Team, checker, index, symbols):
    started = time.time()
    rec = solve(llm, checker, index, symbols, [args.request], args.fix_rounds)[0]
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    # Everything about the attempt, so a build that ran somewhere else (GitHub runner) can be inspected.
    (out / "result.json").write_text(json.dumps({k: v for k, v in rec.items() if k != "first_answer"}, indent=2),
                                     encoding="utf-8")
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


def cmd_evaluate(args, llm: Team, checker, index, symbols):
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


def provider_settings(args) -> tuple[str, str]:
    """(api base, api key) for --provider, with --api-base / NULLIFIED_API_KEY overriding."""
    base, key_env, key_url = PROVIDERS[args.provider]
    key = os.environ.get("NULLIFIED_API_KEY") or (os.environ.get(key_env) if key_env else "")
    if key_env and not key:
        sys.exit(f"Set {key_env} to your free {args.provider} API key (get one at {key_url}). See local/README.md.")
    return (args.api_base or base), key


def cmd_models(args):
    """List the models the provider offers (only the free ones on OpenRouter), biggest context first."""
    base, key = provider_settings(args)
    if not base.endswith("/v1"):  # Ollama: what's downloaded
        with urllib.request.urlopen(base.rstrip("/") + "/api/tags", timeout=30) as resp:
            for m in json.loads(resp.read())["models"]:
                print(f"{m['name']:50} {m['size'] / 1e9:.1f} GB")
        return
    req = urllib.request.Request(base.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        models = json.loads(resp.read())["data"]
    if args.provider == "openrouter":
        models = [m for m in models if m["id"].endswith(":free")]
    models.sort(key=lambda m: -(m.get("context_length") or m.get("context_window") or 0))
    for m in models:
        ctx = m.get("context_length") or m.get("context_window") or "?"
        print(f"{m['id']:60} context {ctx}")
    print(f"\n{len(models)} models. Bigger (e.g. 30B+) coder/instruct models write better mods. "
          "Use one with --model <id> (or set NULLIFIED_MODEL).")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("build", "evaluate", "models"):
        s = sub.add_parser(name)
        s.add_argument("--provider", choices=list(PROVIDERS),
                       default=os.environ.get("NULLIFIED_PROVIDER", "openrouter"),
                       help="openrouter / groq / cerebras: free online API (no disk space). ollama: your own GPU")
        s.add_argument("--api-base", default=os.environ.get("NULLIFIED_API_BASE", ""),
                       help="override the provider URL (e.g. http://localhost:1234/v1 for LM Studio)")
        if name == "models":
            continue
        s.add_argument("--model", default=os.environ.get("NULLIFIED_MODEL", ""),
                       help="model id (run the `models` command to see the free ones; env NULLIFIED_MODEL)")
        s.add_argument("--context", type=int, default=int(os.environ.get("NULLIFIED_CONTEXT", 0)),
                       help="context window in tokens (default: 32768 online, 16384 on an 8 GB GPU)")
        s.add_argument("--parallel", type=int, default=1, help="requests sent to the model at once")
        s.add_argument("--fallback-model", default=os.environ.get("NULLIFIED_FALLBACK_MODEL", "qwen3:8b"),
                       help="Ollama model on your GPU that takes over when the online AI refuses on its own rules "
                            "('none' to turn off)")
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
    if args.command == "models":
        return cmd_models(args)

    base, key = provider_settings(args)
    local = args.provider == "ollama"
    if not args.model:
        if not local:
            sys.exit(f"Pick a model: run `python local/nullified_local.py models --provider {args.provider}` "
                     "and pass one with --model <id> (or set NULLIFIED_MODEL).")
        args.model = "qwen3:8b"
    context = args.context or (16384 if local else 32768)
    main_model = Model(base, args.model, context, args.parallel, key, label=args.model)
    fallback = None
    if not local and args.fallback_model.lower() != "none":
        fallback = Model(PROVIDERS["ollama"][0], args.fallback_model, 16384, 1, label=args.fallback_model)
    llm = Team(main_model, fallback)
    checker = Checker(args.checks, args.ref)
    index, symbols = load_reference(args.reference_dir)
    {"build": cmd_build, "evaluate": cmd_evaluate}[args.command](args, llm, checker, index, symbols)


if __name__ == "__main__":
    main()
