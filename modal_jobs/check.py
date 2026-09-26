"""Quick check that Modal works: runs one tiny CPU job and reports whether the HF secret is set up.

    python -m modal run modal_jobs/check.py
"""
import modal

app = modal.App("nullified-ai-check")


@app.function(cpu=0.25, timeout=60)
def ping() -> str:
    import platform
    return f"cloud job ran on {platform.system()} {platform.machine()}, Python {platform.python_version()}"


@app.local_entrypoint()
def main():
    print(ping.remote())
    try:
        modal.Secret.from_name("huggingface").hydrate()
        print("huggingface secret: found")
    except modal.exception.NotFoundError:
        print("huggingface secret: MISSING (create it in the Modal dashboard: Secrets > Hugging Face, name it 'huggingface')")
