"""Show the fix hints the teacher would get for each failing mod of a run (CPU only)."""
import modal

app = modal.App("nullified-ai-test-hints")
image = modal.Image.debian_slim().pip_install("huggingface_hub").add_local_dir("tools", "/root/nai/tools")


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface")], cpu=1, timeout=300)
def hints(run: str, version: str = "1.21.11", chars: int = 1800) -> str:
    import json, os, sys
    sys.path.insert(0, "/root/nai/tools")
    from huggingface_hub import HfApi, hf_hub_download
    from symbols import SymbolIndex
    token = os.environ["HF_TOKEN"]
    user = HfApi(token=token).whoami()["name"]
    idx = SymbolIndex.load(hf_hub_download(f"{user}/nullified-ai-reference", f"{version}/minecraft-{version}.jsonl", repo_type="dataset", token=token))
    recs = [json.loads(l) for l in open(hf_hub_download(f"{user}/nullified-ai-data", f"raw/{run}.jsonl", repo_type="dataset", token=token), encoding="utf-8")]
    out = []
    for r in recs:
        if r.get("refused"):
            out.append(f"## {r['id']} REFUSED: {r['refused']}")
        if r["errors"]:
            out.append(f"## {r['id']}\n" + idx.hints_for_errors(r["errors"], r["files"])[:chars])
    return "\n\n".join(out)


@app.local_entrypoint()
def main(run: str):
    print(hints.remote(run))
