"""Use Nullified AI: mod request in, .zip + .jar out. Also runs the evaluation.

Build a mod (outputs land in ./output/<modid>/):
    python -m modal run modal_jobs/nullified.py::build --request "a ruby sword that sets mobs on fire"
Evaluate (trained vs. base model on tools/eval_requests.json):
    python -m modal run modal_jobs/nullified.py::evaluate

The pipeline is the specialist team: the model plans + writes the mod, CPU workers compile it,
boot a real 1.21.11 server with it and check its assets, and the model fixes whatever failed.
Requests matching restrictions.txt are refused, and generated code is scanned before anything is built.
"""
import json
import os
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import REPO_DIR, VERSION, compile_batch, compile_image  # noqa: E402
from generate import app as checks_app  # noqa: E402

BASE = "Qwen/Qwen3.5-9B"
ADAPTER_REPO = "nullified-ai-lora"
GPU = "L40S"

app = modal.App("nullified-ai")
app.include(checks_app)
cache = modal.Volume.from_name("nullified-cache", create_if_missing=True)
model_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .pip_install("vllm", "huggingface_hub")
    .env({"HF_HOME": "/cache/hf", "VLLM_LOGGING_LEVEL": "WARNING", "CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir("tools", f"{REPO_DIR}/tools")
    .add_local_file("restrictions.txt", f"{REPO_DIR}/restrictions.txt")
    .add_local_python_source("generate")  # nullified.py imports the shared checks from generate.py
)


@app.function(image=compile_image, cpu=2, memory=6144, timeout=1800)
def package(files: dict[str, str]) -> dict:
    """Full Gradle build of a passing mod -> {"mod_id", "jar": bytes, "zip": bytes}."""
    import shutil
    import subprocess
    sys.path.insert(0, f"{REPO_DIR}/tools")
    from batch_compile import prepare_work_project
    from build_mod import assemble_project, read_mod_id
    from textures import placeholder_textures

    sample = Path("/tmp/sample")
    shutil.rmtree(sample, ignore_errors=True)
    for rel, body in files.items():
        (sample / rel).parent.mkdir(parents=True, exist_ok=True)
        (sample / rel).write_text(body, encoding="utf-8")
    placeholders = placeholder_textures(files)
    for rel, png in placeholders.items():
        (sample / rel).parent.mkdir(parents=True, exist_ok=True)
        (sample / rel).write_bytes(png)
    mod_id = read_mod_id(sample / "src")

    # Build in the pre-warmed project (Minecraft is already downloaded and remapped there).
    work = prepare_work_project(VERSION)
    shutil.rmtree(work / "src", ignore_errors=True)
    shutil.rmtree(work / "build" / "libs", ignore_errors=True)
    shutil.copytree(sample / "src", work / "src")
    subprocess.run(["./gradlew", "build", "--console=plain"], cwd=work, check=True)
    jar = next(j for j in (work / "build" / "libs").glob("*.jar") if not j.name.endswith("-sources.jar"))

    # The .zip is a clean Gradle project named after the mod, ready to open in an IDE.
    project = Path(f"/tmp/zip/{mod_id}")
    assemble_project(sample, VERSION, project, mod_id)
    zip_path = shutil.make_archive(f"/tmp/{mod_id}-project", "zip", project)
    return {"mod_id": mod_id, "jar_name": f"{mod_id}-1.0.0.jar", "jar": jar.read_bytes(),
            "zip": Path(zip_path).read_bytes(), "placeholder_textures": sorted(placeholders)}


@app.function(image=model_image, gpu=GPU, volumes={"/cache": cache}, timeout=3 * 3600,
              secrets=[modal.Secret.from_name("huggingface")])
def solve(requests: list[str], use_adapter: bool = True, fix_rounds: int = 3) -> list[dict]:
    """Run the plan -> write -> check -> fix loop for each request. Returns one record per request."""
    sys.path.insert(0, f"{REPO_DIR}/tools")
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from prompts import (FIX_PROMPT, STAGE_EXPLAINED, STUDENT_SOLVE_PROMPT, format_files, looks_looped,
                         parse_answer, parse_review, review_prompt, runtime_hints, system_prompt)
    from symbols import SymbolIndex
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    token = os.environ["HF_TOKEN"]
    user = HfApi(token=token).whoami()["name"]
    symbols = SymbolIndex.load(hf_hub_download(f"{user}/nullified-ai-reference", f"{VERSION}/minecraft-{VERSION}.jsonl",
                                               repo_type="dataset", token=token))
    lora = None
    if use_adapter:
        path = snapshot_download(f"{user}/{ADAPTER_REPO}", token=token, local_dir="/tmp/adapter")
        lora = LoRARequest("nullified", 1, path)
    llm = LLM(model=BASE, download_dir="/cache/hf", max_model_len=32768, gpu_memory_utilization=0.9,
              enable_lora=use_adapter, max_lora_rank=64, max_num_seqs=64)
    system = system_prompt(VERSION)

    def chat_full(prompts: list[str], temperature: float, max_tokens: int = 14000, repetition_penalty: float = 1.05) -> list:
        convs = [[{"role": "system", "content": system}, {"role": "user", "content": p}] for p in prompts]
        params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens,
                                repetition_penalty=repetition_penalty)
        outs = llm.chat(convs, params, lora_request=lora, chat_template_kwargs={"enable_thinking": False})
        return [o.outputs[0] for o in outs]

    def chat(prompts: list[str], temperature: float, max_tokens: int = 14000) -> list[str]:
        return [o.text for o in chat_full(prompts, temperature, max_tokens)]

    def stuck(out) -> bool:
        return out.finish_reason == "length" or looks_looped(out.text)

    # Restrictions gate: every request is checked against restrictions.txt before anything is built.
    blocked = [parse_review(a) for a in chat([review_prompt(r) for r in requests], 0.0, max_tokens=200)]
    records = [{"id": f"req{i}", "request": req, "plan": "", "files": {}, "refused": rule, "ok": False,
                "stage": "refused" if rule else None, "errors": [], "stage_history": []}
               for i, (req, rule) in enumerate(zip(requests, blocked))]

    allowed = [r for r in records if not r["refused"]]
    solve_prompts = [STUDENT_SOLVE_PROMPT.format(version=VERSION, request=r["request"]) for r in allowed]
    outs = chat_full(solve_prompts, 0.4)
    # Answers that got stuck repeating themselves or ran out of tokens get one retry with more varied sampling.
    retry = [i for i, o in enumerate(outs) if stuck(o)]
    for i, o in zip(retry, chat_full([solve_prompts[i] for i in retry], 0.8, repetition_penalty=1.15)):
        if not stuck(o) or len(parse_answer(o.text)["files"]) > len(parse_answer(outs[i].text)["files"]):
            outs[i] = o
    for r, out in zip(allowed, outs):
        parsed = parse_answer(out.text)
        r.update(plan=parsed["plan"], files=parsed["files"], refused=parsed["refused"],
                 stage="refused" if parsed["refused"] else ("no-files" if not parsed["files"] else None),
                 first_answer=out.text, first_finish=out.finish_reason, first_tokens=len(out.token_ids))

    for rnd in range(fix_rounds + 1):
        todo = [{"id": r["id"], "files": r["files"]} for r in records if r["files"] and not r["ok"]]
        by_id = {r["id"]: r for r in records}
        for batch in compile_batch.map([todo[i:i + 3] for i in range(0, len(todo), 3)]):
            for res in batch:
                by_id[res["id"]].update(ok=res["ok"], stage=res["stage"], errors=res["errors"])
        for r in records:
            r["stage_history"].append(r["stage"])
        failing = [r for r in records if r["files"] and not r["ok"] and r["stage"] in STAGE_EXPLAINED]
        if rnd == fix_rounds or not failing:
            break
        prompts = [FIX_PROMPT.format(version=VERSION, stage=STAGE_EXPLAINED[r["stage"]], request=r["request"],
                                     files=format_files(r["files"]),
                                     errors="\n\n".join(e[:600] for e in r["errors"][:15]),
                                     hints=symbols.hints_for_errors(r["errors"], r["files"])[:20000] + runtime_hints(r["errors"]))
                   for r in failing]
        for r, out in zip(failing, chat_full(prompts, 0.2)):
            if not stuck(out):  # a looping fix would overwrite good files with garbage
                r["files"] = {**r["files"], **parse_answer(out.text)["files"]}
    return records


@app.local_entrypoint()
def build(request: str, fix_rounds: int = 4):
    started = time.time()
    rec = solve.remote([request], True, fix_rounds)[0]
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
            lines = e.splitlines()
            print("  " + lines[0][:200])
            for cause in [l.strip() for l in lines[1:] if l.strip().startswith("Caused by")][-1:]:
                print("    " + cause[:200])  # the root cause of a crash
        return
    out = package.remote(rec["files"])
    folder = Path("output") / out["mod_id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / out["jar_name"]).write_bytes(out["jar"])
    (folder / f"{out['mod_id']}-project.zip").write_bytes(out["zip"])
    print(f"PLAN:\n{rec['plan']}\n\nDone in {(time.time() - started) / 60:.1f} min -> {folder.resolve()}")
    if out["placeholder_textures"]:
        print(f"{len(out['placeholder_textures'])} placeholder textures were added (replace them with real art):")
        for t in out["placeholder_textures"]:
            print("  " + t)


@app.local_entrypoint()
def evaluate(fix_rounds: int = 3, base_too: bool = True, limit: int = 0):
    requests = [e["request"] for e in json.loads(Path("tools/eval_requests.json").read_text())]
    requests = requests[:limit] if limit else requests
    variants = [("nullified", True)] + ([("base", False)] if base_too else [])
    report = {}
    for name, use_adapter in variants:
        recs = solve.remote(requests, use_adapter, fix_rounds)
        rounds = max(len(r["stage_history"]) for r in recs)
        per_round = [sum(1 for r in recs if len(r["stage_history"]) > k and r["stage_history"][k] == "passed"
                         or (len(r["stage_history"]) <= k and r["ok"])) for k in range(rounds)]
        # Every eval request is allowed, so any refusal here is a wrong refusal.
        report[name] = {"first_try": per_round[0], "after_fixes": per_round[-1], "per_round": per_round,
                        "wrongly_refused": sum(bool(r["refused"]) for r in recs), "of": len(recs),
                        "final_stages": {s: sum(r["stage"] == s for r in recs) for s in {r["stage"] for r in recs}},
                        "cut_off": sum(r.get("first_finish") == "length" for r in recs)}
        print(name, json.dumps(report[name]))
        Path("output").mkdir(exist_ok=True)
        with open(f"output/eval_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                print(json.dumps(r, ensure_ascii=False), file=f)
    Path("output/eval.json").write_text(json.dumps(report, indent=2))
