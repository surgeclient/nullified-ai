"""Compile-check a generated run: raw/<run>.jsonl (HF) -> sample folders -> compile -> compiled/<run>.jsonl (HF).

Each output record is the raw record plus {"ok", "stage", "errors", "seconds"}.

Usage (HF_TOKEN in the environment):
    python tools/compile_dataset.py --run smoke1
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_compile import ROOT, check_sample, prepare_work_project  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402


def materialize(record: dict, dest: Path) -> None:
    shutil.rmtree(dest, ignore_errors=True)
    for rel, body in record["files"].items():
        file = dest / rel
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(body, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--template", default="1.21.11")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    repo = f"{api.whoami()['name']}/nullified-ai-data"
    raw = hf_hub_download(repo, f"raw/{args.run}.jsonl", repo_type="dataset", token=token)
    records = [json.loads(line) for line in open(raw, encoding="utf-8")]

    work = prepare_work_project(args.template)
    samples = ROOT / ".work" / "samples" / args.run
    out = ROOT / "results" / f"{args.run}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    passed = checked = 0
    with out.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records, 1):
            if rec["refused"] or not rec["files"]:
                rec.update(ok=False, stage="refused" if rec["refused"] else "no-files", errors=[], seconds=0.0)
            else:
                sample = samples / rec["id"]
                materialize(rec, sample)
                result = check_sample(sample, work, args.timeout)
                rec.update(ok=result["ok"], stage=result["stage"], errors=result["errors"], seconds=result["seconds"])
                checked += 1
                passed += result["ok"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(records)}] {rec['id']} {rec['topic']}: {rec['stage']} {rec['seconds']}s", flush=True)

    api.upload_file(path_or_fileobj=str(out), path_in_repo=f"compiled/{args.run}.jsonl",
                    repo_id=repo, repo_type="dataset")
    print(f"\n{passed}/{checked} compiled ({len(records)} records). Uploaded compiled/{args.run}.jsonl")


if __name__ == "__main__":
    main()
