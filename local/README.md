# Nullified AI without Modal

## Recommended: everything on GitHub

The model (gpt-oss-20b, later your own trained one) runs on GitHub's free servers, together with the compiling,
the Minecraft test server and the packaging (`.github/workflows/nullified.yml`). Your PC only starts the job and
downloads the `.jar`. This needs the repo to be **public**: public repos get 16 GB RAM servers and unlimited free
minutes.

### Set it up (once)
1. **Make the repo public:** in nullified-ai go to **Settings → General**, scroll to the bottom (Danger Zone), click
   **Change visibility → Make public**, and confirm. The code history was checked, and no keys are in it.
2. **Add your Hugging Face token as a secret:** go to **Settings → Secrets and variables → Actions → New repository
   secret**, name it `HF_TOKEN`, and paste a Hugging Face token with **Read** access as the value. Secrets are never
   shown in the code or logs, not even in a public repo, and people without write access can't run your workflows.
3. **On your PC:** your `GITHUB_TOKEN` needs **Actions: Read and write** and **Contents: Read** on nullified-ai.
   Never put a key in a file in this repo; keys belong only in `setx` and GitHub Secrets.

### Build a mod
- **From the website:** go to **Actions → nullified → Run workflow**, type the mod you want and click **Run
  workflow**. When the run finishes (about 45–90 min), download **mod** under **Artifacts** at the bottom of the run page.
- **From your PC:**
  ```
  python local/nullified_remote.py --request "a ruby sword that sets mobs on fire"
  ```
  This downloads the result into `output/`.

### Good to know
- **Anyone can see what you build.** Because the repo is public, anyone can see the code, `restrictions.txt`, the
  requests, the logs and the built mods. Your secrets and your private Hugging Face model and datasets stay hidden.
- **It's slow.** The model runs on a CPU, so one attempt takes about 15–25 minutes. Several builds can run at the same time.
- **To use your own trained model later:** pass `--model-repo <you>/<repo> --model-file <file>.gguf`.

## Other option: online API for the model



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
