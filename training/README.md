# Training your own Nullified AI model (free)

Two steps: **gather data** (overnight, on GitHub), then **train** (Colab, free GPU).

## 1. Gather training data (already automated)
Run the `teach` workflow (website Training tab, or Actions -> teach). Teacher models write mods, every mod is
compiled and booted on a real 1.21.11 server, and only the results are saved to your private Hugging Face dataset
`nullified-ai-data`. Run it over several nights to build up a few hundred to a few thousand passing mods.

## 2. Train (free GPU)

### Kaggle — free P100 16 GB (use this for gpt-oss-20B)
gpt-oss must train in **float32** (no bf16/fp16 support for it on these cards), so the 20B model does **not** fit
Colab's free T4 (14.5 GB) — it OOMs by a few dozen MB no matter how you trim it. Kaggle's free **P100 (16 GB)** has
just enough headroom.
1. Kaggle -> **Create -> New Notebook**, then **File -> Import Notebook** and upload
   `training/nullified_train_kaggle.ipynb` from this repo (or paste the one-cell block below).
2. Right panel: **Accelerator = GPU P100**, **Internet = On** (one-time phone verification).
3. **Run All.** Paste your Hugging Face **Write** token when asked. ~30-60 min; saves the adapter and a GGUF.

One-cell fallback (paste into a single Kaggle cell, P100 + Internet on):
```python
!pip install -q unsloth 'huggingface_hub>=0.26'
!git clone -q https://github.com/surgeclient/nullified-ai.git /kaggle/working/nullified-ai || (cd /kaggle/working/nullified-ai && git pull -q)
import os, getpass
os.environ['HF_TOKEN'] = getpass.getpass('HF WRITE token: ').strip()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
!cd /kaggle/working/nullified-ai && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python training/train_lora.py --runs repos3 --adapter-repo nullified-coder --epochs 3 --max-seq-train 4096 --rank 16 --attn-only --gguf
```

### Colab — free T4 (only for smaller bases)
`training/nullified_train.ipynb` works on Colab's free T4 for a smaller base model (pass `--base` to a 7B-class
coder). It will **not** fit gpt-oss-20B — use Kaggle above for that.

4. Use your model: on the website's Build tab, **Options -> Model** = `<you>/nullified-coder-gguf`.

Why not GitHub: training needs a GPU; GitHub's free servers are CPU-only. Kaggle/Colab free GPUs are the no-cost
option, but they need you to click Run in the browser.

## Specialists (later)
`--only build` trains a coder, `--only fix` trains a fixer, each to its own `--adapter-repo`. The coordinator can
then load the base once and swap adapters. Start with one coder, prove it beats plain gpt-oss-20b, then add more.
