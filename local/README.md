# Run Nullified AI on your own GPU (free, no Modal)

- **The model** runs on your GPU through [Ollama](https://ollama.com). This is the only thing stored on your PC (about 5 GB).
- **Compiling, booting a real 1.21.11 server and building the .jar/.zip** run on GitHub Actions
  (`.github/workflows/check-mod.yml`). Your PC doesn't need Gradle, Minecraft or Java. Only the finished
  `.jar` and project `.zip` are downloaded into `output/<modid>/`.

## One-time setup

1. **Install Ollama** from https://ollama.com/download. Before pulling a model, you can move where models are stored
   (for example to a drive with more space) by setting the `OLLAMA_MODELS` environment variable.
2. **Download a model that fits your GPU.** For an 8 GB GPU:
   ```
   ollama pull qwen3:8b
   ```
   | GPU VRAM | Model | Disk |
   |---|---|---|
   | 6–8 GB | `qwen3:8b` (default) | ~5 GB |
   | 12 GB | `qwen3:14b` | ~9 GB |
   | 16–24 GB | `qwen3:30b` or bigger | ~18 GB |

   If VRAM is tight, set `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` before starting Ollama.
   This roughly halves the memory used by the 16k context.
3. **Make a GitHub token** so the script can start the check workflow: GitHub → Settings → Developer settings →
   Fine-grained tokens → Generate. Give it access to **only this repository**, with **Actions: Read and write**
   and **Contents: Read**. Then set it in your terminal:
   ```
   setx GITHUB_TOKEN "github_pat_..."        (Windows; open a new terminal afterwards)
   export GITHUB_TOKEN=github_pat_...        (Mac/Linux)
   ```
4. *(Recommended)* Set `HF_TOKEN` too (and run `pip install huggingface_hub`). The script then downloads the 1.21.11
   reference docs from your private `nullified-ai-reference` dataset and gives them to the model. Small models
   get much better results with them.

`check-mod.yml` must be on the repo's **default branch** (GitHub only runs manually triggered workflows from
there). Merge this branch first.

## Use it

Python 3.10+ is all you need. Keep Ollama running.
```
python local/nullified_local.py build --request "a ruby sword that sets mobs on fire"
python local/nullified_local.py evaluate --limit 5
```
Useful options: `--model qwen3:14b`, `--context 24576` (if your GPU has room), `--fix-rounds 6`,
`--checks local` (compile on your PC instead; needs Java 21 and about 3 GB more).
LM Studio or llama.cpp work too: pass their OpenAI-style URL, e.g. `--api-base http://localhost:1234/v1`.

## Good to know

- Each round of checks is one GitHub Actions run of about 3–6 minutes (the first run takes longer while it
  caches Minecraft). Public repos get unlimited free minutes. Private repos on the free plan get 2,000 minutes
  a month.
- The LoRA adapter trained by `modal_jobs/train.py` is for the 27B model, so it can't be used with a small
  local model. The local run uses the plain model plus the same prompts, reference docs, checks and fix loop.
- `modal_jobs/` (data generation and training) still runs on Modal. Training a 27B model needs far more
  GPU memory than 8 GB.
