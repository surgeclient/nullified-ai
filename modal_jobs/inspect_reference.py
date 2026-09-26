"""Spot-check the private reference dataset from Modal (also proves the HF token works there).

    python -m modal run modal_jobs/inspect_reference.py
"""
import modal

app = modal.App("nullified-ai-inspect")
image = modal.Image.debian_slim().pip_install("huggingface_hub")


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface")], cpu=1, timeout=300)
def inspect(version: str = "1.21.11") -> str:
    import json
    import os
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=os.environ["HF_TOKEN"])
    repo = f"{api.whoami()['name']}/nullified-ai-reference"
    lines = [f"dataset: {repo}"]

    def load(name):
        path = hf_hub_download(repo, f"{version}/{name}", repo_type="dataset", token=os.environ["HF_TOKEN"])
        return [json.loads(l) for l in open(path, encoding="utf-8")]

    mc = load(f"minecraft-{version}.jsonl")
    ident = next(r for r in mc if r["path"] == "net.minecraft.resources.Identifier")
    lines.append(f"minecraft signatures: {len(mc)}; Identifier record:\n" + "\n".join(ident["text"].splitlines()[:8]))

    docs = load(f"docs-{version}.jsonl")
    titles = sorted({r["title"].split(" > ")[0] for r in docs if r["kind"] == "doc"})
    examples = [r for r in docs if r["kind"] == "example"]
    lines.append(f"docs: {len(docs)} ({len(examples)} example files); sample pages: {titles[:12]}")
    translated = [r["path"] for r in docs if "/translated/" in r["path"]]
    lines.append(f"translated pages mixed in: {len(translated)}")

    api_recs = load(f"fabric-api-{version}.jsonl")
    lines.append(f"fabric api: {len(api_recs)}; e.g. {[r['title'] for r in api_recs[:3]]}")
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    print(inspect.remote())
