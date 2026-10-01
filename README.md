# Nullified AI

A team of specialist AI models (one base model + LoRA adapters) trained to write
Fabric Minecraft mods for 1.21.11 and newer.

## Web page
Build mods and start training data from https://surgeclient.github.io/nullified-ai/ (`docs/index.html`, served by
GitHub Pages: Settings → Pages → Deploy from a branch → `main` / `/docs`). Your GitHub token stays in your browser.

## Run it free (no Modal)
Everything on GitHub (model included): see [`local/README.md`](local/README.md), then
`python local/nullified_remote.py --request "..."`. Or use a free online API: See [`local/README.md`](local/README.md). The model runs on a free online API (30B+ models, no disk space) with
a small fallback on your own GPU. Compiling, testing and packaging run on GitHub Actions. Only
`restrictions.txt` decides what gets refused.
```
python local/nullified_local.py models
python local/nullified_local.py build --model <id> --request "a ruby sword that sets mobs on fire"
```

## Layout
| Path | What it is |
|---|---|
| `restrictions.txt` | Everything the AI refuses. One rule per line. Anything not listed is allowed. |
| `templates/1.21.11/` | Official Fabric template (Mojang mappings) that every mod is built on |
| `samples/` | Mod samples (each is a `src/` folder). `samples/smoke/` holds pipeline tests |
| `tools/batch_compile.py` | Compile-checks many samples with one warm Gradle setup |
| `tools/build_mod.py` | Builds one sample into a `.jar` + project `.zip` |
| `local/nullified_local.py` | Runs the whole pipeline without Modal (free online model + GPU fallback, checks on GitHub Actions) |
| `local/nullified_remote.py` | Starts a build on GitHub (`nullified` workflow: model + checks on GitHub's servers) and downloads the jar |
| `tools/mod_checks.py` | Compile + real-server + asset checks and packaging, used by the `check-mod` workflow |
| `tools/restrictions.py` | Loads `restrictions.txt` and scans code for harmful patterns |
| `.github/workflows/` | `compile-check` (batch), `build-mod` (jar + zip), `collect-reference` (corpus → HF), `check-mod` (used by the local runner), `nullified` (whole build incl. model); all run manually |

## Versions (1.21.11 template)
Minecraft 1.21.11 · Fabric Loader 0.19.5 · Fabric API 0.141.6+1.21.11 · Mojang mappings

## Roadmap
1. ✅ Repo, restrictions, 1.21.11 template, compile + build workflows
2. ✅ Reference corpus (`collect-reference` workflow → private HF dataset `nullified-ai-reference`)
3. Data generator (open-weight teacher run on Modal with vLLM) + compile/restriction filter
4. Modal training: base model + specialist adapters (Planner, Coder, Mixin, Assets, Fixer, Reviewer)
5. Evaluation suite
6. Request queue: request → specialists → build → `.zip` + `.jar`, with auto-fix on build errors
7. Docs lookup for 26.x and newer

## Secrets
Stored in GitHub Secrets / Modal Secrets only, never in code: `HF_TOKEN`.
