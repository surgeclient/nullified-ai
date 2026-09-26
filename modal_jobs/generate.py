"""Generate compile-verified training mods on Modal.

  1. Teacher (vLLM on one GPU) writes mod requests for each topic x difficulty.
  2. Teacher writes PLAN + all files for each request, grounded in 1.21.11 reference material + cheat sheet.
  3. CPU workers (warm Gradle + Loom, many in parallel) compile every mod against the real 1.21.11 template.
  4. Failures go back to the teacher with the compiler errors + real signatures of the classes involved.
     Repeats up to --fix-rounds times. Every attempt is kept (Fixer training data).
Output: raw/<run>.jsonl in the private HF dataset <you>/nullified-ai-data.

Smoke test:  python -m modal run modal_jobs/generate.py::main --run smoke2 --topics items,blocks,commands --per-combo 2 --difficulties 1
Full run:    python -m modal run modal_jobs/generate.py::main --run gen1 --per-combo 25
"""
import json
import os
import random
import sys
import time

import modal

TEACHER = "Qwen/Qwen3.8-27B-FP8"
VERSION = "1.21.11"
GPU = "H100"
GPU_USD_PER_HOUR = 3.95
REPO_DIR = "/root/nai"  # tools/, restrictions.txt, templates/ live here in every container

app = modal.App("nullified-ai-generate")
cache = modal.Volume.from_name("nullified-cache", create_if_missing=True)

teacher_image = (
    # CUDA devel image: vLLM JIT-compiles some kernels at startup and needs nvcc + headers.
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .pip_install("vllm", "huggingface_hub")
    .env({"HF_HOME": "/cache/hf", "VLLM_LOGGING_LEVEL": "WARNING", "CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir("tools", f"{REPO_DIR}/tools")
    .add_local_file("restrictions.txt", f"{REPO_DIR}/restrictions.txt")
)

# Compile workers: JDK + the template with Minecraft already downloaded/remapped at image build time,
# so each worker compiles a mod in seconds.
compile_image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("openjdk-21-jdk-headless")
    .apt_install("procps")  # pkill, to stop a hung test server
    .add_local_dir(f"templates/{VERSION}", f"{REPO_DIR}/templates/{VERSION}", copy=True)
    .add_local_dir("tools/probe", "/root/probe_src", copy=True)
    .add_local_file("tools/runtime_check.py", "/root/runtime_check.py", copy=True)
    .run_commands(
        f"mkdir -p {REPO_DIR}/.work && cp -r {REPO_DIR}/templates/{VERSION} {REPO_DIR}/.work/{VERSION}",
        f"cd {REPO_DIR}/.work/{VERSION} && chmod +x gradlew && ./gradlew compileJava compileClientJava --console=plain --no-daemon",
        # Build the runtime probe mod against the same template.
        f"cp -r {REPO_DIR}/templates/{VERSION} /root/probe_build && rm -rf /root/probe_build/src"
        " && cp -r /root/probe_src/src /root/probe_build/src",
        "cd /root/probe_build && chmod +x gradlew && ./gradlew build --console=plain --no-daemon",
        f"cp $(ls /root/probe_build/build/libs/*.jar | grep -v sources) {REPO_DIR}/probe.jar",
        # Warm-up server boot with the template mod; also proves the probe works (fails the build if not).
        f"cd /root && python -c \"import sys; sys.path.insert(0, '/root'); from pathlib import Path; "
        f"from runtime_check import run_server; ok, errs, probe = run_server(Path('{REPO_DIR}/.work/{VERSION}'), "
        f"Path('{REPO_DIR}/probe.jar'), timeout=900); print(ok, errs, probe); sys.exit(0 if ok else 1)\"",
    )
    .add_local_dir("tools", f"{REPO_DIR}/tools")
    .add_local_file("restrictions.txt", f"{REPO_DIR}/restrictions.txt")
)


@app.function(image=compile_image, cpu=2, memory=6144, timeout=3600, max_containers=60)
def compile_batch(items: list[dict]) -> list[dict]:
    """Compile each mod, boot a real server with it, validate its assets.
    items: [{"id", "files"}] -> [{"id", "ok", "stage", "errors", "seconds"}]
    stage: compile | runtime | assets | restrictions | json | timeout | passed"""
    import shutil
    from pathlib import Path
    sys.path.insert(0, f"{REPO_DIR}/tools")
    from batch_compile import check_sample, prepare_work_project
    from runtime_check import run_server, validate_assets

    work = prepare_work_project(VERSION)
    results = []
    for item in items:
        sample = Path(f"/tmp/samples/{item['id']}")
        shutil.rmtree(sample, ignore_errors=True)
        for rel, body in item["files"].items():
            (sample / rel).parent.mkdir(parents=True, exist_ok=True)
            (sample / rel).write_text(body, encoding="utf-8")
        start = time.time()
        r = check_sample(sample, work, timeout=300)
        ok, stage, errors = r["ok"], r["stage"], r["errors"]
        if ok:
            ok, errors, probe = run_server(work, Path(f"{REPO_DIR}/probe.jar"))
            stage = "runtime"
            if ok:
                errors = validate_assets(item["files"], probe)
                ok, stage = not errors, "assets" if errors else "passed"
        results.append({"id": item["id"], "ok": ok, "stage": stage, "errors": errors,
                        "seconds": round(time.time() - start, 1)})
    return results


@app.function(image=teacher_image, volumes={"/cache": cache}, timeout=3600, cpu=2,
              secrets=[modal.Secret.from_name("huggingface")])
def prefetch_teacher() -> str:
    """Download the teacher weights on a cheap CPU container so the GPU never waits on downloads."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(TEACHER, cache_dir="/cache/hf", token=os.environ["HF_TOKEN"])
    cache.commit()
    return path


def compile_all(records: list[dict], chunk: int = 3) -> None:
    """Compile every record that has files and hasn't passed yet; write results onto the records."""
    todo = [{"id": r["id"], "files": r["files"]} for r in records if r["files"] and not r.get("ok")]
    by_id = {r["id"]: r for r in records}
    chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
    for batch in compile_batch.map(chunks):
        for res in batch:
            by_id[res["id"]].update(ok=res["ok"], stage=res["stage"], errors=res["errors"])


@app.function(image=teacher_image, gpu=GPU, volumes={"/cache": cache}, timeout=6 * 3600,
              secrets=[modal.Secret.from_name("huggingface")])
def generate(run: str, topics: list[str], difficulties: list[int], per_combo: int,
             fix_rounds: int = 3, seed: int = 0, thinking: bool = False) -> dict:
    sys.path.insert(0, f"{REPO_DIR}/tools")
    from huggingface_hub import HfApi, snapshot_download
    from prompts import (DIFFICULTY, FIX_PROMPT, REQUEST_PROMPT, SOLVE_PROMPT, STAGE_EXPLAINED, TOPICS,
                         format_files, parse_answer, runtime_hints, system_prompt)
    from retrieval import ReferenceIndex
    from symbols import SymbolIndex
    from vllm import LLM, SamplingParams

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    user = api.whoami()["name"]
    started = time.time()

    ref_dir = snapshot_download(f"{user}/nullified-ai-reference", repo_type="dataset", token=token,
                                allow_patterns=[f"{VERSION}/*"], local_dir="/tmp/reference")
    index = ReferenceIndex.load(f"{ref_dir}/{VERSION}", VERSION)
    symbols = SymbolIndex.load(f"{ref_dir}/{VERSION}/minecraft-{VERSION}.jsonl")

    llm = LLM(model=TEACHER, max_model_len=32768, gpu_memory_utilization=0.92,
              download_dir="/cache/hf", seed=seed,
              max_num_seqs=256)  # hybrid (Mamba) model: one cache block per sequence
    chat_kwargs = {"enable_thinking": thinking}
    system = system_prompt(VERSION)

    def chat(user_prompts: list[str], temperature: float, max_tokens: int) -> list:
        convs = [[{"role": "system", "content": system}, {"role": "user", "content": p}] for p in user_prompts]
        return llm.chat(convs, SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens, seed=seed),
                        chat_template_kwargs=chat_kwargs)

    # --- Stage 1: requests ---
    combos = [(t, d) for t in topics for d in difficulties]
    outs = chat([REQUEST_PROMPT.format(n=per_combo, topic=t, topic_hint=TOPICS[t], difficulty=DIFFICULTY[d])
                 for t, d in combos], temperature=0.9, max_tokens=3000)
    jobs = []
    for (topic, diff), out in zip(combos, outs):
        text = out.outputs[0].text
        try:
            reqs = json.loads(text[text.index("["):text.rindex("]") + 1])
        except ValueError:
            continue
        jobs += [{"topic": topic, "difficulty": diff, "request": r.strip()} for r in reqs[:per_combo]
                 if isinstance(r, str) and len(r.strip()) > 15]
    random.Random(seed).shuffle(jobs)
    print(f"stage 1: {len(jobs)} requests from {len(combos)} combos", flush=True)

    # --- Stage 2: first solutions ---
    prompts = [SOLVE_PROMPT.format(version=VERSION, request=j["request"],
                                   reference=index.context_for(f"{j['topic']} {j['request']}", budget_chars=36000))
               for j in jobs]
    outs = chat(prompts, temperature=0.6, max_tokens=14000)
    records, out_tokens = [], 0
    for i, (job, out) in enumerate(zip(jobs, outs)):
        parsed = parse_answer(out.outputs[0].text)
        out_tokens += len(out.outputs[0].token_ids)
        records.append({
            "id": f"{run}-{i:05d}", "run": run, "version": VERSION, "teacher": TEACHER,
            "topic": job["topic"], "difficulty": job["difficulty"], "request": job["request"],
            "plan": parsed["plan"], "files": parsed["files"], "refused": parsed["refused"],
            "finish": out.outputs[0].finish_reason, "ok": False, "stage": None, "errors": [],
            "attempts": [],  # each failed compile + the teacher's fix (Fixer training data)
        })

    # --- Stage 3/4: compile, then fix what failed ---
    history = []
    for rnd in range(fix_rounds + 1):
        compile_all(records)
        passed = sum(r["ok"] for r in records)
        history.append(passed)
        stages = {}
        for r in records:
            stages[r["stage"]] = stages.get(r["stage"], 0) + 1
        print(f"round {rnd}: {passed}/{len(records)} pass all checks {stages}", flush=True)
        failing = [r for r in records if r["files"] and not r["ok"] and r["stage"] in STAGE_EXPLAINED]
        if rnd == fix_rounds or not failing:
            break
        fix_prompts = []
        for r in failing:
            errors = "\n\n".join(e[:600] for e in r["errors"][:15])
            hints = symbols.hints_for_errors(r["errors"], r["files"])[:20000]
            fix_prompts.append(FIX_PROMPT.format(version=VERSION, request=r["request"], files=format_files(r["files"]),
                                                 stage=STAGE_EXPLAINED[r["stage"]], errors=errors,
                                                 hints=hints + runtime_hints(r["errors"])))
        outs = chat(fix_prompts, temperature=0.3, max_tokens=14000)
        for r, out in zip(failing, outs):
            out_tokens += len(out.outputs[0].token_ids)
            changed = parse_answer(out.outputs[0].text)["files"]
            r["attempts"].append({"round": rnd, "stage": r["stage"], "errors": r["errors"],
                                  "files_before": dict(r["files"]), "changed": changed})
            r["files"] = {**r["files"], **changed}

    path = f"/tmp/{run}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    repo = f"{user}/nullified-ai-data"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=path, path_in_repo=f"raw/{run}.jsonl", repo_id=repo, repo_type="dataset")

    minutes = (time.time() - started) / 60
    return {
        "run": run, "requests": len(records), "passed_per_round": history,
        "passed": sum(r["ok"] for r in records), "refused": sum(bool(r["refused"]) for r in records),
        "cut_off": sum(r["finish"] == "length" for r in records), "output_tokens": out_tokens,
        "gpu_minutes": round(minutes, 1), "approx_gpu_cost_usd": round(minutes / 60 * GPU_USD_PER_HOUR, 2),
        "saved_to": f"{repo}/raw/{run}.jsonl",
    }


@app.local_entrypoint()
def main(run: str, topics: str = "", difficulties: str = "1,2,3", per_combo: int = 10,
         fix_rounds: int = 3, seed: int = 0, thinking: bool = False):
    sys.path.insert(0, "tools")
    from prompts import TOPICS
    topic_list = [t.strip() for t in topics.split(",") if t.strip()] or list(TOPICS)
    unknown = [t for t in topic_list if t not in TOPICS]
    if unknown:
        raise SystemExit(f"unknown topics: {unknown}. Choose from: {', '.join(TOPICS)}")
    diffs = [int(d) for d in difficulties.split(",")]
    print(f"run={run}: {len(topic_list)} topics x {len(diffs)} difficulties x {per_combo} = "
          f"up to {len(topic_list) * len(diffs) * per_combo} mods on {GPU} with {TEACHER}, {fix_rounds} fix rounds")
    print("teacher weights:", prefetch_teacher.remote())
    print(json.dumps(generate.remote(run, topic_list, diffs, per_combo, fix_rounds, seed, thinking), indent=2))


@app.local_entrypoint()
def test_compile():
    """CPU-only check of the compile workers using samples/smoke (expects everything but outdated_api to pass)."""
    from pathlib import Path
    items = []
    for sample in sorted(Path("samples/smoke").iterdir()):
        files = {p.relative_to(sample).as_posix(): p.read_text(encoding="utf-8")
                 for p in sample.rglob("*") if p.is_file() and p.suffix in {".java", ".json"}}
        items.append({"id": sample.name, "files": files})
    for res in compile_batch.remote(items):
        print(f"{res['id']}: {'PASS' if res['ok'] else 'FAIL'} ({res['stage']}, {res['seconds']}s)")
        for e in res["errors"][:2]:
            print("   ", e.splitlines()[0][-150:])
