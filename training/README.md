# Training your own Nullified AI model (free)

Two steps: **gather data** (overnight, on GitHub), then **train** (Colab, free GPU).

## 1. Gather training data (already automated)
Run the `teach` workflow (website Training tab, or Actions -> teach). Teacher models write mods, every mod is
compiled and booted on a real 1.21.11 server, and only the results are saved to your private Hugging Face dataset
`nullified-ai-data`. Run it over several nights to build up a few hundred to a few thousand passing mods.

## 2. Train (Google Colab, free T4 GPU)
1. Open `training/nullified_train.ipynb` in Colab (github.com -> the file -> "Open in Colab", or upload it).
2. **Runtime -> Change runtime type -> T4 GPU.**
3. **Runtime -> Run all.** Paste your Hugging Face **Write** token when asked, set the run names, and wait
   (about 1-3 hours). It saves the adapter and a GGUF to your Hugging Face.
4. Use your model: on the website's Build tab, **Options -> Model** = `<you>/nullified-coder-gguf`.

Why Colab and not GitHub: training needs a GPU; GitHub's free servers are CPU-only. Colab's free GPU is the
no-cost option, but it needs you to click Run (it signs into your Google account in the browser).

## Specialists (later)
`--only build` trains a coder, `--only fix` trains a fixer, each to its own `--adapter-repo`. The coordinator can
then load the base once and swap adapters. Start with one coder, prove it beats plain gpt-oss-20b, then add more.
