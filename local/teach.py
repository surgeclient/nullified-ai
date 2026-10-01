"""Make training data with teacher models from OpenRouter (or Groq / Cerebras): no Modal, no GPU.

Teachers invent mod requests, write each mod with the 1.21.11 reference docs, the mod is compiled and booted on a
real server (Java on this machine - the teach workflow runs it on GitHub), and failures go back to the teacher with
the errors. Every record (passing mods + each fix attempt) is saved in the same format as modal_jobs/generate.py,
so tools/build_sft.py and the trainer can use it. Results go to <you>/nullified-ai-data/raw/<run>.jsonl (private).

    python local/teach.py --run night1 --teachers "openrouter:<model id>,openrouter:<other id>" --minutes 330
Keys: OPENROUTER_API_KEY / GROQ_API_KEY / CEREBRAS_API_KEY for the providers you use, HF_TOKEN to upload.
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "local"), str(ROOT / "tools")]
from nullified_local import (PROVIDERS, VERSION, Model, ModelUnavailable, fix_view, load_reference,  # noqa: E402
                             self_refused, stuck)
from prompts import (DIFFICULTY, FIX_PROMPT, REQUEST_PROMPT, SOLVE_PROMPT, STAGE_EXPLAINED, TOPICS,  # noqa: E402
                     format_files, parse_answer, runtime_hints)
from runtime_check import fix_data_dirs  # noqa: E402


class OutOfQuota(Exception):
    pass


# Teacher families whose licenses allow training on their outputs (check each model page to be sure).
ALLOWED = re.compile(r"qwen|deepseek|nemotron|mistral|mixtral|devstral|codestral|glm|kimi|minimax", re.I)
SKIP = re.compile(r"safety|guard|nano|embed|rerank|vision|-vl|ocr|audio|tts|whisper", re.I)


def model_size(model_id: str) -> float:
    """Parameter count in billions from the name (e.g. ...-550b-a55b -> 550); 0 when the name doesn't say."""
    sizes = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)b(?![a-z])", model_id.lower())]
    return max(sizes) if sizes else 0.0


def big_enough(model_id: str) -> bool:
    """Skip models that are clearly small (the name says under 20B); unknown sizes are allowed."""
    size = model_size(model_id)
    return size == 0 or size >= 20


class Teachers:
    """Round-robin over teacher models. A teacher that keeps failing is dropped and, when possible, replaced by
    another allowed model from the same provider; when a provider's whole daily quota is used up, the run
    continues with the other providers you have keys for."""

    def __init__(self, specs: list[str], context: int, auto: bool = True):
        self.context, self.auto = context, auto
        self.models, self.lock, self.i = [], threading.Lock(), 0
        self.tried, self.exhausted, self.fails = set(), set(), {}
        for spec in specs:
            provider, _, model = spec.partition(":")
            if provider not in PROVIDERS or not model:
                sys.exit(f"teacher must look like provider:model-id, got {spec!r}")
            self._add(provider, model)
        if auto:  # every other provider you have a key for joins as a backup
            for provider, (_, key_env, _) in PROVIDERS.items():
                if key_env and os.environ.get(key_env) and not any(m.provider == provider for m in self.models):
                    self._refill(provider)
        if not self.models:
            sys.exit("no usable teachers (check the model ids and that the API key secrets are set)")
        print("teachers: " + ", ".join(f"{m.provider}:{m.model}" for m in self.models), flush=True)

    def _add(self, provider: str, model: str) -> bool:
        base, key_env, key_url = PROVIDERS[provider]
        key = os.environ.get(key_env, "") if key_env else ""
        if key_env and not key:
            print(f"skipping {provider}:{model}: {key_env} is not set ({key_url})")
            return False
        self.tried.add((provider, model))
        m = Model(base, model, self.context, 1, key, label=f"{provider}:{model}")
        m.provider = provider
        self.models.append(m)
        return True

    def _refill(self, provider: str) -> None:
        """Add the biggest-context allowed model from this provider that hasn't been tried yet."""
        if provider in self.exhausted or provider in ("ollama", "server"):
            return
        base, key_env, _ = PROVIDERS[provider]
        try:
            req = urllib.request.Request(base.rstrip("/") + "/models",
                                         headers={"Authorization": f"Bearer {os.environ.get(key_env, '')}"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                listed = json.loads(resp.read())["data"]
        except Exception as e:
            print(f"  could not list {provider} models: {str(e)[:150]}", flush=True)
            return
        if provider == "openrouter":
            listed = [m for m in listed if m["id"].endswith(":free")]
        # Biggest model first (by the size in its name, e.g. 550b), then the longest context.
        listed.sort(key=lambda m: (-model_size(m["id"]), -(m.get("context_length") or m.get("context_window") or 0)))
        for m in listed:
            mid = m["id"]
            if (provider, mid) not in self.tried and ALLOWED.search(mid) and not SKIP.search(mid) and big_enough(mid):
                print(f"  adding teacher {provider}:{mid}", flush=True)
                self._add(provider, mid)
                return

    def chat(self, prompt: str, temperature: float, max_tokens: int = 14000) -> tuple[dict, str]:
        while True:
            with self.lock:
                if not self.models:
                    raise OutOfQuota("every teacher is out of quota")
                model = self.models[self.i % len(self.models)]
                self.i += 1
            try:
                out = model.chat([prompt], temperature, max_tokens=max_tokens)[0], model.label
                self.fails[model.label] = 0  # it worked, reset its failure streak
                return out
            except ModelUnavailable as e:
                msg = str(e)
                # ModelUnavailable is only raised for HTTP/network errors, never bad content, so it's always the
                # teacher's fault, not the prompt's.
                self.fails[model.label] = self.fails.get(model.label, 0) + 1
                dead = ("daily limit" in msg or "error 401" in msg)  # account done for the day, or bad key
                overloaded = self.fails[model.label] >= 4  # keeps 500/502/503/504-ing: stop wasting time on it
                if not (dead or overloaded):
                    continue  # transient - let another teacher take the next turn, try this one again later
                print(f"  teacher {model.label} dropped ({msg[:150]})", flush=True)
                with self.lock:
                    if model not in self.models:
                        continue
                    self.models.remove(model)
                    if dead:
                        self.exhausted.add(model.provider)
                        self.models = [m for m in self.models if m.provider != model.provider]
                    if self.auto:
                        self._refill(model.provider)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="name of this batch, e.g. night1")
    p.add_argument("--teachers", default=os.environ.get("NULLIFIED_TEACHERS", ""),
                   help="comma-separated provider:model-id (run `nullified_local.py models` for free ids)")
    p.add_argument("--topics", default="", help="comma-separated (default: all)")
    p.add_argument("--difficulties", default="1,2,3")
    p.add_argument("--per-combo", type=int, default=6, help="requests per topic x difficulty")
    p.add_argument("--fix-rounds", type=int, default=3)
    p.add_argument("--parallel", type=int, default=3, help="mods worked on at once (checks still run one by one)")
    p.add_argument("--minutes", type=float, default=330, help="stop starting new mods after this long")
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-auto", action="store_true",
                   help="only use the listed teachers (default: replace ones that run out, add other providers)")
    p.add_argument("--require-hf", action="store_true", help="stop at once if the data can't be saved to Hugging Face")
    args = p.parse_args()

    started = time.time()
    deadline = started + args.minutes * 60
    teachers = Teachers([t.strip() for t in args.teachers.split(",") if t.strip()], args.context, not args.no_auto)
    index, symbols = load_reference("")
    import mod_checks  # Java + Gradle on this machine
    check_lock = threading.Lock()

    out = ROOT / "output" / "data"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.run}.jsonl"
    records, save_lock = [], threading.Lock()
    hf = None
    if os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ["HF_TOKEN"])
            repo = f"{api.whoami()['name']}/nullified-ai-data"
            api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
            hf = (api, repo)
        except Exception as e:
            print(f"WARNING: can't upload to Hugging Face ({str(e).splitlines()[-1][:200]}); saving locally only")
    if args.require_hf and not hf:
        sys.exit("HF_TOKEN must be a valid Hugging Face token with WRITE access, so the data can be saved.")
    last_upload = [0.0]

    def save(force: bool = False):
        with save_lock:
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
            if hf and (force or time.time() - last_upload[0] > 600):  # upload every 10 min so a cut-off loses little
                try:
                    hf[0].upload_file(path_or_fileobj=str(path), path_in_repo=f"raw/{args.run}.jsonl",
                                      repo_id=hf[1], repo_type="dataset", commit_message=f"{args.run}: {len(records)} records")
                    last_upload[0] = time.time()
                except Exception as e:
                    print(f"  upload failed (will retry): {str(e)[:150]}", flush=True)

    # --- 1. requests ---
    topics = [t.strip() for t in args.topics.split(",") if t.strip()] or list(TOPICS)
    combos = [(t, int(d)) for t in topics for d in args.difficulties.split(",")]
    random.Random(args.seed).shuffle(combos)
    jobs = []

    def make_requests(combo):
        topic, diff = combo
        try:
            out, _ = teachers.chat(REQUEST_PROMPT.format(n=args.per_combo, topic=topic, topic_hint=TOPICS[topic],
                                                         difficulty=DIFFICULTY[diff]), 0.9, max_tokens=3000)
            text = out["text"]
            reqs = json.loads(text[text.index("["):text.rindex("]") + 1])
        except (ValueError, OutOfQuota, ModelUnavailable):
            return []
        return [{"topic": topic, "difficulty": diff, "request": r.strip()} for r in reqs[:args.per_combo]
                if isinstance(r, str) and len(r.strip()) > 15]

    with ThreadPoolExecutor(args.parallel) as pool:
        for got in pool.map(make_requests, combos):
            jobs += got
    random.Random(args.seed).shuffle(jobs)
    print(f"{len(jobs)} requests from {len(combos)} topic/difficulty combos", flush=True)

    # --- 2-4. write, check, fix - one mod at a time per worker ---
    def check(rec):
        with check_lock:
            rec["files"] = fix_data_dirs(rec["files"])
            res = mod_checks.check(rec["id"], rec["files"])
        rec.update(ok=res["ok"], stage=res["stage"], errors=res["errors"])

    def work(i_job):
        i, job = i_job
        if time.time() > deadline:
            return
        rec = {"id": f"{args.run}-{i:05d}", "run": args.run, "version": VERSION, "teacher": None,
               "topic": job["topic"], "difficulty": job["difficulty"], "request": job["request"],
               "plan": "", "files": {}, "refused": None, "finish": None, "ok": False, "stage": None,
               "errors": [], "attempts": []}
        try:
            prompt = SOLVE_PROMPT.format(version=VERSION, request=job["request"],
                                         reference=index.context_for(f"{job['topic']} {job['request']}",
                                                                     budget_chars=36000))
            out, teacher = teachers.chat(prompt, 0.6)
            if stuck(out):
                out, teacher = teachers.chat(prompt, 0.8)
            if self_refused(out):
                return  # the teacher's own rules, not ours - not useful as training data
            parsed = parse_answer(out["text"])
            rec.update(teacher=teacher, plan=parsed["plan"], files=parsed["files"], refused=parsed["refused"],
                       finish=out["finish"])
            if not rec["files"]:
                return
            for rnd in range(args.fix_rounds + 1):
                check(rec)
                if rec["ok"] or rnd == args.fix_rounds or rec["stage"] not in STAGE_EXPLAINED:
                    break
                fix = FIX_PROMPT.format(version=VERSION, stage=STAGE_EXPLAINED[rec["stage"]], request=rec["request"],
                                        files=format_files(fix_view(rec["files"], rec["stage"], rec["errors"])),
                                        errors="\n\n".join(e[:600] for e in rec["errors"][:15]),
                                        hints=symbols.hints_for_errors(rec["errors"], rec["files"])[:20000]
                                        + runtime_hints(rec["errors"]))
                out, _ = teachers.chat(fix, 0.3)
                if stuck(out) or self_refused(out):
                    break
                changed = parse_answer(out["text"])["files"]
                rec["attempts"].append({"round": rnd, "stage": rec["stage"], "errors": rec["errors"],
                                        "files_before": dict(rec["files"]), "changed": changed})
                rec["files"] = {**rec["files"], **changed}
        except OutOfQuota:
            pass
        except Exception as e:  # one bad mod must not stop the night's run
            print(f"  skipped {rec['id']}: {str(e).splitlines()[0][:200] if str(e) else repr(e)}", flush=True)
        finally:
            if rec["files"]:
                with save_lock:
                    records.append(rec)
                    n, ok = len(records), sum(r["ok"] for r in records)
                print(f"[{n}] {'PASS' if rec['ok'] else 'fail (' + str(rec['stage']) + ')'} {rec['request'][:80]}"
                      f"  -> {ok} passing so far", flush=True)
                save()

    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(work, enumerate(jobs)))
    save(force=True)

    passed = sum(r["ok"] for r in records)
    fixes = sum(len(r["attempts"]) for r in records)
    summary = {"run": args.run, "records": len(records), "passing_mods": passed, "fix_attempts": fixes,
               "minutes": round((time.time() - started) / 60, 1), "saved_to": f"{hf[1]}/raw/{args.run}.jsonl" if hf
               else str(path), "teachers_left": [m.model for m in teachers.models]}
    print(json.dumps(summary, indent=2))
    (out / f"{args.run}.summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
