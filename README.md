# Nullified AI

A team of specialist AI models (one base model + LoRA adapters) trained to write
Fabric Minecraft mods for 1.21.11 and newer.

## Layout
| Path | What it is |
|---|---|
| `restrictions.txt` | Everything the AI refuses. One rule per line. Anything not listed is allowed. |
| `templates/1.21.11/` | Official Fabric template (Mojang mappings) that every mod is built on |
| `samples/` | Mod samples (each is a `src/` folder). `samples/smoke/` holds pipeline tests |
| `tools/batch_compile.py` | Compile-checks many samples with one warm Gradle setup |
| `tools/build_mod.py` | Builds one sample into a `.jar` + project `.zip` |
| `tools/restrictions.py` | Loads `restrictions.txt` and scans code for harmful patterns |
| `.github/workflows/` | `compile-check` (batch) and `build-mod` (jar + zip), both run manually |

## Versions (1.21.11 template)
Minecraft 1.21.11 · Fabric Loader 0.19.5 · Fabric API 0.141.6+1.21.11 · Mojang mappings

## Roadmap
1. ✅ Repo, restrictions, 1.21.11 template, compile + build workflows
2. ✅ Reference corpus (`collect-reference` workflow → private HF dataset `nullified-ai-reference`)
3. Data generator (free open-weight teachers via OpenRouter) + compile/restriction filter
4. Kaggle training: base model + specialist adapters (Planner, Coder, Mixin, Assets, Fixer, Reviewer)
5. Evaluation suite
6. Request queue: request → specialists → build → `.zip` + `.jar`, with auto-fix on build errors
7. Docs lookup for 26.x and newer

## Secrets
Stored in GitHub Secrets / Kaggle Secrets only, never in code: `HF_TOKEN`, `OPENROUTER_API_KEY`.
