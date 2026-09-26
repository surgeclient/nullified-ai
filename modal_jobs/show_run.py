"""Print a summary of a compile-checked run (errors per mod) from the private HF dataset.

    python -m modal run modal_jobs/show_run.py --run smoke1
    python -m modal run modal_jobs/show_run.py --run smoke1 --show-files smoke1-00002
"""
import modal

app = modal.App("nullified-ai-show-run")
image = modal.Image.debian_slim().pip_install("huggingface_hub")


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface")], cpu=1, timeout=300)
def show(run: str, show_files: str = "", max_errors: int = 6) -> str:
    import collections
    import json
    import os
    import re
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ["HF_TOKEN"]
    repo = f"{HfApi(token=token).whoami()['name']}/nullified-ai-data"
    path = hf_hub_download(repo, f"compiled/{run}.jsonl", repo_type="dataset", token=token)
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]

    out = [f"{sum(r['ok'] for r in recs)}/{len(recs)} compiled"]
    symbols = collections.Counter()
    for r in recs:
        out.append(f"\n## {r['id']} [{r['topic']}] {r['stage']} - {r['request'][:110]}")
        for e in r["errors"][:max_errors]:
            short = re.sub(r"^.*?/src/", "src/", e.replace("\n", " | "))
            out.append("  " + short[:300])
            m = re.search(r"symbol:\s+(?:class|method|variable)\s+([\w<>(),. ]+)", e)
            if m:
                symbols[m.group(1).strip()] += 1
        if r["id"] == show_files:
            for p, body in r["files"].items():
                out.append(f"\n--- {p} ---\n{body}")
    out.append("\nmost common missing symbols: " + ", ".join(f"{s} x{n}" for s, n in symbols.most_common(15)))
    return "\n".join(out)


@app.local_entrypoint()
def main(run: str, show_files: str = ""):
    print(show.remote(run, show_files))
