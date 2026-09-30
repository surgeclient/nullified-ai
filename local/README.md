# Run Nullified AI free, without Modal

- **The model** is a big (30B+) model on a free online API: OpenRouter (default), Groq or Cerebras. It uses no
  disk space. If you'd rather use your own GPU, use `--provider ollama`.
- **Compiling, booting a real 1.21.11 server and building the .jar/.zip** run on GitHub Actions
  (`.github/workflows/check-mod.yml`). Your PC doesn't need Gradle, Minecraft or Java. Only the finished `.jar` and
  project `.zip` are downloaded into `output/<modid>/`.
- **Only `restrictions.txt` decides what gets refused.** A block from the online AI only counts if it matches a
  line in `restrictions.txt`. If the online AI refuses because of its own rules, the request goes to a small
  fallback model on your GPU (`qwen3:8b` via Ollama, about 5 GB). That model follows only `restrictions.txt`. The
  code scanner in `tools/restrictions.py` still checks every mod before it's built.

## One-time setup

1. **Get a free API key**: OpenRouter at https://openrouter.ai/keys (or Groq at https://console.groq.com/keys, or
   Cerebras at https://cloud.cerebras.ai). Then set it:
   ```
   setx OPENROUTER_API_KEY "sk-or-..."       (Windows; open a new terminal afterwards)
   export OPENROUTER_API_KEY=sk-or-...       (Mac/Linux)
   ```
   For Groq use `GROQ_API_KEY` with `--provider groq`, and for Cerebras use `CEREBRAS_API_KEY` with `--provider cerebras`.
2. **Pick a model.** This lists the free models, biggest context first:
   ```
   python local/nullified_local.py models
   ```
   Choose a big coder or instruct model (30B or more) and save it so you don't have to type it every time:
   `setx NULLIFIED_MODEL "<model id>"`.
3. **Make a GitHub token** so the script can start the check workflow: GitHub → Settings → Developer settings →
   Fine-grained tokens → Generate. Give it access to **only this repository**, with **Actions: Read and write** and
   **Contents: Read**. Then run `setx GITHUB_TOKEN "github_pat_..."`.
4. *(Recommended)* **Install the fallback model**: install [Ollama](https://ollama.com/download), then run
   `ollama pull qwen3:8b` (about 5 GB). It's only used when the online AI refuses by its own rules. Without it,
   those requests are reported as "refused by the online AI's own rules". To turn the fallback off, use
   `--fallback-model none`.
5. *(Recommended)* **Set `HF_TOKEN`** and run `pip install huggingface_hub`. The model then gets the 1.21.11
   reference docs from your private `nullified-ai-reference` dataset (tens of MB).

`check-mod.yml` must be on the repo's **default branch**, because GitHub only runs manually triggered workflows
from there. Merge this branch first.

## Use it

Python 3.10+ is all you need.
```
python local/nullified_local.py build --request "a ruby sword that sets mobs on fire"
python local/nullified_local.py evaluate --limit 5
```
Useful options: `--model <id>`, `--provider groq`, `--fix-rounds 6`, `--parallel 3` (for evaluate),
`--provider ollama --model qwen3:8b` (everything on your GPU), and `--checks local` (compile on your PC; needs
Java 21 and about 3 GB).

## Good to know

- **Free tiers have rate limits** (requests per minute or per day). The script waits and retries automatically
  when it hits one. A single build uses roughly 2–10 requests.
- **What free providers do with your prompts** depends on each provider's policy, and some free models may log
  them. Your requests and the generated code are sent to them.
- **Check speed:** each round of checks is one GitHub Actions run of about 3–6 minutes (the first run takes
  longer while it caches Minecraft). Public repos get unlimited free minutes. Private repos on the free plan get
  2,000 minutes a month.
- **Your trained adapter isn't used.** The LoRA adapter from `modal_jobs/train.py` only works with the 27B base
  model, so online models run with the same prompts, reference docs, checks and fix loop, but without it.
- **Modal is still used for training.** `modal_jobs/` (data generation and training) still runs on Modal.
