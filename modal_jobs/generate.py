"""Generate training mods with an open-weight teacher running on Modal (vLLM).

Stage 1: the teacher writes realistic mod requests for each topic x difficulty.
Stage 2: for each request, it writes a PLAN + all mod files, grounded in 1.21.11 reference material.
Output: raw/<run>.jsonl in the private HF dataset <you>/nullified-ai-data (compile-checked later on GitHub).

Smoke test (~6 mods, roughly $1):
    python -m modal run modal_jobs/generate.py --run smoke1 --topics items,blocks,commands --per-combo 2 --difficulties 1
Full run:
    python -m modal run modal_jobs/generate.py --run gen1 --per-combo 25
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

app = modal.App("nullified-ai-generate")
cache = modal.Volume.from_name("nullified-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "huggingface_hub")
    .env({"HF_HOME": "/cache/hf", "VLLM_LOGGING_LEVEL": "WARNING"})
    .add_local_dir("tools", "/root/tools")
    .add_local_file("restrictions.txt", "/root/restrictions.txt")
)


def hf_user(api) -> str:
    return api.whoami()["name"]


@app.function(image=image, gpu=GPU, volumes={"/cache": cache}, timeout=6 * 3600,
              secrets=[modal.Secret.from_name("huggingface")])
def generate(run: str, topics: list[str], difficulties: list[int], per_combo: int,
             seed: int = 0, thinking: bool = False) -> dict:
    sys.path.insert(0, "/root/tools")
    from huggingface_hub import HfApi, snapshot_download
    from prompts import (DIFFICULTY, REQUEST_PROMPT, SOLVE_PROMPT, TOPICS, parse_answer, system_prompt)
    from retrieval import ReferenceIndex
    from vllm import LLM, SamplingParams

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    user = hf_user(api)
    started = time.time()

    ref_dir = snapshot_download(f"{user}/nullified-ai-reference", repo_type="dataset", token=token,
                                allow_patterns=[f"{VERSION}/*"], local_dir="/tmp/reference")
    index = ReferenceIndex.load(f"{ref_dir}/{VERSION}", VERSION)

    llm = LLM(model=TEACHER, max_model_len=32768, gpu_memory_utilization=0.92,
              download_dir="/cache/hf", seed=seed)
    cache.commit()  # keep downloaded weights for next time
    chat_kwargs = {"enable_thinking": thinking}
    system = system_prompt(VERSION)

    # --- Stage 1: requests ---
    combos = [(t, d) for t in topics for d in difficulties]
    req_convs = [[{"role": "system", "content": system},
                  {"role": "user", "content": REQUEST_PROMPT.format(
                      n=per_combo, topic=t, topic_hint=TOPICS[t], difficulty=DIFFICULTY[d])}]
                 for t, d in combos]
    outs = llm.chat(req_convs, SamplingParams(temperature=0.9, top_p=0.95, max_tokens=3000, seed=seed),
                    chat_template_kwargs=chat_kwargs)
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
    print(f"stage 1: {len(jobs)} requests from {len(combos)} topic/difficulty combos", flush=True)

    # --- Stage 2: solutions ---
    solve_convs = []
    for job in jobs:
        job["reference"] = index.context_for(f"{job['topic']} {job['request']}", budget_chars=40000)
        solve_convs.append([{"role": "system", "content": system},
                            {"role": "user", "content": SOLVE_PROMPT.format(
                                version=VERSION, request=job["request"], reference=job["reference"])}])
    outs = llm.chat(solve_convs, SamplingParams(temperature=0.6, top_p=0.95, max_tokens=14000, seed=seed),
                    chat_template_kwargs=chat_kwargs)

    records, out_tokens = [], 0
    for i, (job, out) in enumerate(zip(jobs, outs)):
        answer = out.outputs[0].text
        out_tokens += len(out.outputs[0].token_ids)
        parsed = parse_answer(answer)
        records.append({
            "id": f"{run}-{i:05d}", "run": run, "version": VERSION, "teacher": TEACHER,
            "topic": job["topic"], "difficulty": job["difficulty"], "request": job["request"],
            "plan": parsed["plan"], "files": parsed["files"], "refused": parsed["refused"],
            "dropped": parsed["dropped"], "finish": out.outputs[0].finish_reason, "raw": answer,
        })

    path = f"/tmp/{run}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    repo = f"{user}/nullified-ai-data"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=path, path_in_repo=f"raw/{run}.jsonl", repo_id=repo, repo_type="dataset")

    minutes = (time.time() - started) / 60
    return {
        "run": run, "requests": len(jobs), "with_files": sum(bool(r["files"]) for r in records),
        "refused": sum(bool(r["refused"]) for r in records),
        "cut_off": sum(r["finish"] == "length" for r in records),
        "output_tokens": out_tokens, "gpu_minutes": round(minutes, 1),
        "approx_cost_usd": round(minutes / 60 * 3.95, 2), "saved_to": f"{repo}/raw/{run}.jsonl",
    }


@app.local_entrypoint()
def main(run: str, topics: str = "", difficulties: str = "1,2,3", per_combo: int = 10,
         seed: int = 0, thinking: bool = False):
    sys.path.insert(0, "tools")
    from prompts import TOPICS
    topic_list = [t.strip() for t in topics.split(",") if t.strip()] or list(TOPICS)
    unknown = [t for t in topic_list if t not in TOPICS]
    if unknown:
        raise SystemExit(f"unknown topics: {unknown}. Choose from: {', '.join(TOPICS)}")
    diffs = [int(d) for d in difficulties.split(",")]
    print(f"run={run}: {len(topic_list)} topics x {len(diffs)} difficulties x {per_combo} = "
          f"up to {len(topic_list) * len(diffs) * per_combo} mods on {GPU} with {TEACHER}")
    print(json.dumps(generate.remote(run, topic_list, diffs, per_combo, seed, thinking), indent=2))
