"""Train a Nullified AI LoRA adapter from the teacher data. Runs on any CUDA GPU (free Colab T4 works).

Uses Unsloth so gpt-oss-20b fits a 16 GB GPU. Builds the same chat examples as modal_jobs (tools/build_sft.py),
trains a LoRA, saves it, and (with a Write HF_TOKEN) uploads it to <you>/<adapter-repo> plus a GGUF the runner can serve.

    python training/train_lora.py --runs night-2026-10-01-p1,night-2026-10-01-p2 --adapter-repo nullified-coder
Colab: see training/nullified_train.ipynb (one click).
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
VERSION = "1.21.11"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="comma-separated run names in <you>/nullified-ai-data/raw/<run>.jsonl")
    p.add_argument("--base", default="unsloth/gpt-oss-20b", help="base model (an Unsloth build loads fastest)")
    p.add_argument("--adapter-repo", default="nullified-coder", help="HF model repo to upload the adapter to")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--max-length", type=int, default=16384)
    p.add_argument("--max-seq-train", type=int, default=0, help="override context if VRAM is tight (e.g. 8192)")
    p.add_argument("--only", choices=["build", "fix", "both"], default="both", help="which example kinds to train on")
    p.add_argument("--gguf", action="store_true", help="also export a merged GGUF (needed to serve on the runner)")
    p.add_argument("--data-repo", default="", help="HF dataset (default <you>/nullified-ai-data)")
    p.add_argument("--dry-run", action="store_true", help="build the dataset and stop (no GPU needed)")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi(token=token)
    user = api.whoami()["name"]
    data_repo = args.data_repo or f"{user}/nullified-ai-data"

    # --- gather the teacher records ---
    records = []
    for run in [r.strip() for r in args.runs.split(",") if r.strip()]:
        path = hf_hub_download(data_repo, f"raw/{run}.jsonl", repo_type="dataset", token=token)
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        records += rows
        print(f"{run}: {len(rows)} records, {sum(r.get('ok') for r in rows)} passing", flush=True)
    if not records:
        sys.exit("no records found - check --runs and that the overnight run saved data")

    from build_sft import build_examples, fix_examples
    from symbols import SymbolIndex
    examples = []
    if args.only in ("build", "both"):
        examples += build_examples(records, VERSION)
    # Fix examples need the symbol reference AND records that actually have failed attempts (teacher data).
    # Real collected mods have no attempts, so skip the reference download entirely when it isn't needed.
    want_fix = args.only in ("fix", "both") and any(r.get("attempts") for r in records)
    if want_fix:
        try:
            ref = snapshot_download(f"{user}/nullified-ai-reference", repo_type="dataset", token=token,
                                    allow_patterns=[f"{VERSION}/*"])
            symbols = SymbolIndex.load(f"{ref}/{VERSION}/minecraft-{VERSION}.jsonl")
            examples += fix_examples(records, VERSION, symbols)
        except Exception as e:
            print(f"(skipping fix examples; reference unavailable: {str(e)[:150]})", flush=True)
    random.Random(0).shuffle(examples)
    kinds = {k: sum(e["kind"] == k for e in examples) for k in ("build", "fix")}
    print(f"{len(examples)} training examples {kinds}", flush=True)
    if len(examples) < 20:
        print("WARNING: very few examples - the adapter will barely change the model. Gather more teacher data first.")
    if args.dry_run:
        return

    # --- load the base model with Unsloth (fits gpt-oss-20b on a 16 GB GPU) ---
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    max_seq = args.max_seq_train or args.max_length
    model, tokenizer = FastLanguageModel.from_pretrained(args.base, max_seq_length=max_seq, load_in_4bit=True,
                                                         dtype=None, full_finetuning=False)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=0)

    def to_text(e):
        return tokenizer.apply_chat_template(e["messages"], tokenize=False, add_generation_prompt=False)

    from datasets import Dataset
    rows = [{"text": to_text(e)} for e in examples]
    # keep examples that fit the training window
    rows = [r for r in rows if len(tokenizer(r["text"]).input_ids) <= max_seq]
    print(f"kept {len(rows)}/{len(examples)} examples within {max_seq} tokens", flush=True)
    dataset = Dataset.from_list(rows)

    from trl import SFTConfig, SFTTrainer
    cfg = SFTConfig(output_dir="/tmp/out", per_device_train_batch_size=1, gradient_accumulation_steps=8,
                    warmup_ratio=0.05, num_train_epochs=args.epochs, learning_rate=args.lr, logging_steps=5,
                    optim="adamw_8bit", lr_scheduler_type="cosine", seed=0, report_to=[], max_length=max_seq)
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=cfg)
    # train only on the assistant's answers, not the prompt (standard for instruction tuning)
    try:
        trainer = train_on_responses_only(trainer, instruction_part="<|start|>user<|message|>",
                                          response_part="<|start|>assistant<|message|>")
    except Exception as e:
        print(f"(training on full sequence; response-only masking unavailable: {str(e)[:120]})")
    trainer.train()

    out = Path("/tmp/adapter")
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "nullified_manifest.json").write_text(json.dumps(
        {"base": args.base, "version": VERSION, "runs": args.runs, "examples": len(rows), "kinds": kinds,
         "epochs": args.epochs, "lr": args.lr, "rank": args.rank}, indent=2))

    repo = f"{user}/{args.adapter_repo}"
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo, repo_type="model", commit_message=f"train on {args.runs}")
    print(f"adapter uploaded -> {repo}", flush=True)

    if args.gguf:
        # Merge the adapter into the base and export GGUF so the GitHub runner (llama.cpp) can serve it.
        gguf_repo = f"{repo}-gguf"
        try:
            model.push_to_hub_gguf(gguf_repo, tokenizer, quantization_method="q4_k_m", token=token)
            print(f"GGUF uploaded -> {gguf_repo}  (use it on the runner: --model-repo {gguf_repo})", flush=True)
        except Exception as e:
            print(f"GGUF export failed ({str(e)[:200]}). The adapter is still saved; export can be retried.")

    print("\nDONE. Next: build with your model using the runner's advanced options:")
    print(f"  model repo = {repo}-gguf   (or run training again with --gguf if you skipped it)")


if __name__ == "__main__":
    main()
