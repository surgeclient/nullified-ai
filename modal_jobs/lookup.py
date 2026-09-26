"""Look up real 1.21.11 Minecraft classes/methods (used to verify cheat-sheet facts).

    python -m modal run modal_jobs/lookup.py --queries "BlockPos;SwordItem;net.minecraft.world.item.Item$Properties:sword"
A query is either a simple class name (where does it live?) or FQN:filter (methods containing filter).
"""
import modal

app = modal.App("nullified-ai-lookup")
image = modal.Image.debian_slim().pip_install("huggingface_hub").add_local_dir("tools", "/root/tools")


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface")], cpu=1, timeout=300)
def lookup(queries: list[str], version: str = "1.21.11") -> str:
    import os
    import sys
    sys.path.insert(0, "/root/tools")
    from huggingface_hub import HfApi, hf_hub_download
    from symbols import SymbolIndex

    token = os.environ["HF_TOKEN"]
    repo = f"{HfApi(token=token).whoami()['name']}/nullified-ai-reference"
    idx = SymbolIndex.load(hf_hub_download(repo, f"{version}/minecraft-{version}.jsonl", repo_type="dataset", token=token))
    out = []
    for q in queries:
        if ":" in q:
            fqn, flt = q.split(":", 1)
            ms = idx.methods(fqn, flt)
            out.append(f"{fqn} [{flt}]: " + ("\n    " + "\n    ".join(ms[:60]) if ms else "(no matching methods / class missing)"))
        else:
            out.append(f"{q}: {idx.find_class(q) or 'DOES NOT EXIST'}")
    return "\n".join(out)


@app.local_entrypoint()
def main(queries: str):
    print(lookup.remote([q.strip() for q in queries.split(";") if q.strip()]))
