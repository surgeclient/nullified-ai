"""Download a .gguf model from Hugging Face and print its local path (used by the nullified workflow).

    python tools/fetch_model.py --repo ggml-org/gpt-oss-20b-GGUF [--file name.gguf] --dir /mnt/nai/model
Without --file (or if that file doesn't exist) it picks one: MXFP4, then Q4_K_M, then any Q4, then the smallest.
Split models (-00001-of-0000N) are downloaded in full; the first part is printed.
"""
import argparse
import os
import re
import sys

from huggingface_hub import HfApi, hf_hub_download

PREFER = [r"mxfp4", r"q4_k_m", r"q4", r""]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--file", default="")
    p.add_argument("--dir", required=True)
    args = p.parse_args()
    token = os.environ.get("HF_TOKEN") or None

    try:
        info = HfApi().model_info(args.repo, files_metadata=True, token=token)
    except Exception as e:  # an invalid HF_TOKEN would also block public models
        if token is None:
            raise
        print(f"HF_TOKEN rejected ({str(e).splitlines()[-1][:150]}); trying without it", file=sys.stderr)
        token = None
        info = HfApi().model_info(args.repo, files_metadata=True, token=False)
    infos = {f.rfilename: f.size or 0 for f in info.siblings if f.rfilename.endswith(".gguf")}
    print(f"{args.repo}: {sorted(infos)}", file=sys.stderr)
    if not infos:
        sys.exit(f"no .gguf files in {args.repo}")
    if args.file in infos:
        pick = args.file
    else:
        if args.file:
            print(f"{args.file} not found, choosing automatically", file=sys.stderr)
        # one entry per model: the first shard of split files, whole files otherwise; skip vision projectors
        firsts = [f for f in infos if not re.search(r"-0000[2-9]-of-|mmproj", f, re.I)]
        pick = None
        for pattern in PREFER:
            matches = [f for f in firsts if re.search(pattern, f, re.I)]
            if matches:
                pick = min(matches, key=lambda f: infos[f])
                break
    shard = re.search(r"-00001-of-(\d+)", pick)
    parts = [pick.replace("-00001-of-", f"-{i:05d}-of-") for i in range(1, int(shard.group(1)) + 1)] if shard else [pick]
    print(f"downloading {parts} ({sum(infos.get(x, 0) for x in parts) / 1e9:.1f} GB)", file=sys.stderr)
    paths = [hf_hub_download(args.repo, part, local_dir=args.dir, token=token if token else False) for part in parts]
    print(paths[0])


if __name__ == "__main__":
    main()
