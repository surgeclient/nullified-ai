"""Fine-tune Nullified AI (LoRA on an open-weight base) from verified generator runs.

    python -m modal run modal_jobs/train.py --runs med1,gen1
Output: private HF model repo <you>/nullified-ai-lora (adapter + tokenizer + training manifest).
"""
import json
import os
import sys
import time

import modal

BASE = "Qwen/Qwen3.5-9B"
VERSION = "1.21.11"
GPU = "H100"
GPU_USD_PER_HOUR = 3.95
REPO_DIR = "/root/nai"

app = modal.App("nullified-ai-train")
cache = modal.Volume.from_name("nullified-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch", "transformers", "peft", "trl", "accelerate", "datasets", "huggingface_hub",
                 "flash-linear-attention", "ninja", "packaging", "wheel")
    # Qwen3.5's linear-attention layers fall back to a much slower PyTorch path without this kernel.
    # Compiled for H100 (sm_90) at image build time, which runs on CPU.
    .apt_install("build-essential", "git")
    .env({"TORCH_CUDA_ARCH_LIST": "9.0", "MAX_JOBS": "8", "CXX": "g++", "CC": "gcc"})
    .pip_install("causal-conv1d", extra_options="--no-build-isolation")
    .env({"HF_HOME": "/cache/hf", "CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir("tools", f"{REPO_DIR}/tools")
    .add_local_file("restrictions.txt", f"{REPO_DIR}/restrictions.txt")
)


def load_base(name: str):
    import torch
    import transformers
    kwargs = {"dtype": torch.bfloat16, "cache_dir": "/cache/hf"}
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except (ValueError, KeyError):
        # Qwen3.5 checkpoints are multimodal; fall back to the image-text class and train its language model.
        return transformers.AutoModelForImageTextToText.from_pretrained(name, **kwargs)


GRAD_ACCUM = 8


def make_configs(n_examples: int, epochs: float, lr: float, rank: int, max_length: int, gpu: bool = True,
                 max_steps: int = -1):
    from peft import LoraConfig
    from trl import SFTConfig
    steps = max(1, int(n_examples * epochs / GRAD_ACCUM))
    config = SFTConfig(
        output_dir="/tmp/out", num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, steps // 20), per_device_train_batch_size=1, gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True, bf16=gpu, use_cpu=not gpu, max_length=max_length,
        logging_steps=5 if max_steps <= 0 else 1,
        save_strategy="no", report_to=[], completion_only_loss=True,
        max_steps=max_steps,
    )
    # Language-model projections only (Qwen3.5 checkpoints also contain a vision tower). These module
    # names are the ones vLLM can serve LoRA on, so the adapter loads at use time.
    lora = LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
    return config, lora


@app.function(image=image, cpu=2, timeout=600)
def validate_configs() -> str:
    """CPU-only check that the training settings are accepted by the installed library versions."""
    import peft
    import transformers
    import trl
    make_configs(100, 3.0, 1e-4, 32, 16384, gpu=False)
    return f"configs OK (transformers {transformers.__version__}, trl {trl.__version__}, peft {peft.__version__})"


@app.function(image=image, volumes={"/cache": cache}, timeout=3600, cpu=2,
              secrets=[modal.Secret.from_name("huggingface")])
def prefetch_base() -> str:
    """Download the base weights on a cheap CPU container so the GPU never waits on downloads."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(BASE, cache_dir="/cache/hf", token=os.environ["HF_TOKEN"])
    cache.commit()
    return path


@app.function(image=image, gpu=GPU, volumes={"/cache": cache}, timeout=4 * 3600,
              secrets=[modal.Secret.from_name("huggingface")])
def train(runs: list[str], epochs: float = 3.0, lr: float = 1e-4, rank: int = 32, max_length: int = 16384,
          dry_run: bool = False, benchmark_steps: int = 0) -> dict:
    sys.path.insert(0, f"{REPO_DIR}/tools")
    import random
    from datasets import Dataset
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer, TrainerCallback
    from trl import SFTTrainer
    from build_sft import build_examples, fix_examples
    from symbols import SymbolIndex

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    user = api.whoami()["name"]
    started = time.time()

    symbols = SymbolIndex.load(hf_hub_download(f"{user}/nullified-ai-reference", f"{VERSION}/minecraft-{VERSION}.jsonl",
                                               repo_type="dataset", token=token))
    records = []
    for run in runs:
        path = hf_hub_download(f"{user}/nullified-ai-data", f"raw/{run}.jsonl", repo_type="dataset", token=token)
        records += [json.loads(l) for l in open(path, encoding="utf-8")]
    examples = build_examples(records, VERSION) + fix_examples(records, VERSION, symbols)
    random.Random(0).shuffle(examples)
    kinds = {k: sum(e["kind"] == k for e in examples) for k in ("build", "fix")}
    print(f"{len(records)} records -> {len(examples)} examples {kinds}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE, cache_dir="/cache/hf")
    rows = [{"prompt": e["messages"][:2], "completion": e["messages"][2:]} for e in examples]
    texts = [tokenizer.apply_chat_template(r["prompt"] + r["completion"], tokenize=False) for r in rows]
    lengths = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
    keep = [r for r, n in zip(rows, lengths) if n <= max_length]
    total_tokens = sum(n for n in lengths if n <= max_length)
    print(f"kept {len(keep)}/{len(rows)} examples <= {max_length} tokens; {total_tokens} tokens per epoch", flush=True)
    if dry_run or not keep:
        return {"examples": len(examples), "kept": len(keep), "kinds": kinds, "tokens_per_epoch": total_tokens}

    model = load_base(BASE)
    cache.commit()
    if benchmark_steps:
        keep = keep * max(1, benchmark_steps * GRAD_ACCUM // len(keep) + 1)  # enough data for the steps
    config, lora = make_configs(len(keep), epochs, lr, rank, max_length, max_steps=benchmark_steps or -1)
    step_times = []

    class StepTimer(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            step_times.append(time.time())

    trainer = SFTTrainer(model=model, args=config, train_dataset=Dataset.from_list(keep),
                         processing_class=tokenizer, peft_config=lora, callbacks=[StepTimer()])
    trainer.model.print_trainable_parameters()
    result = trainer.train()
    if benchmark_steps:
        # Skip the first steps (kernel compilation / autotuning) and report the steady state.
        gaps = [b - a for a, b in zip(step_times, step_times[1:])][1:]
        sec_per_step = sum(gaps) / max(1, len(gaps))
        tokens_per_step = total_tokens / max(1, len(examples)) * GRAD_ACCUM
        return {"steps": len(step_times), "sec_per_step_steady": round(sec_per_step, 1),
                "tokens_per_sec_steady": round(tokens_per_step / max(sec_per_step, 1e-6)),
                "step_gaps": [round(g, 1) for g in gaps]}

    trainer.model.save_pretrained("/tmp/adapter")
    tokenizer.save_pretrained("/tmp/adapter")
    manifest = {"base": BASE, "version": VERSION, "runs": runs, "examples": len(keep), "kinds": kinds,
                "epochs": epochs, "lr": lr, "rank": rank, "final_loss": result.training_loss}
    with open("/tmp/adapter/nullified_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    repo = f"{user}/nullified-ai-lora"
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path="/tmp/adapter", repo_id=repo, repo_type="model",
                      commit_message=f"Train on {', '.join(runs)}")

    minutes = (time.time() - started) / 60
    return {**manifest, "repo": repo, "gpu_minutes": round(minutes, 1),
            "approx_gpu_cost_usd": round(minutes / 60 * GPU_USD_PER_HOUR, 2)}


@app.local_entrypoint()
def main(runs: str, epochs: float = 3.0, lr: float = 1e-4, rank: int = 32, dry_run: bool = False,
         benchmark_steps: int = 0):
    print(validate_configs.remote())  # fail on CPU, not after paying for a GPU
    print("base weights:", prefetch_base.remote())
    print(json.dumps(train.remote([r.strip() for r in runs.split(",") if r.strip()], epochs, lr, rank,
                                  dry_run=dry_run, benchmark_steps=benchmark_steps), indent=2))
