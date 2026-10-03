# Nullified AI - roadmap (captured from planning, 2026-10)

Priority order: get the coding model trained and working first. Everything below layers on after that.

## Now / in progress
- [x] Run without Modal: model + checks + packaging on GitHub's free servers; website to drive it.
- [x] Overnight teacher data (`teach` workflow) -> private HF dataset `nullified-ai-data`.
- [ ] Train the first coding model (LoRA on gpt-oss-20b) from that data, on free Colab. Test, repeat until it
      reliably builds mods that need custom behavior (the copper-hammer case the base model fails).

## Model lineup (fine-tunes of open models, not trained from scratch)
- Nullified 1.0 - 20B, the main coder.
- Nullified Light - ~10B, faster, smaller jobs.
- Nullified Light+ - ~4B, fast Q&A / quick checks (the "Haiku" of the set).
- Specialist LoRA adapters on one base (planner / coder / fixer / reviewer) - cheap, swappable, fit in memory.
  Preferred over separate full models (three 20B models can't run at once in 16 GB).

## Pipeline
- Planner routes a request: coding-only -> coding models only; needs art/video -> bring those in too.
- Big models do the heavy parts, small models the small parts (sub-agents).
- Parallel review: several 4B models run/check the code at once, collect ALL errors, hand the list to the big
  model to fix in ONE pass, then re-test - instead of slow back-and-forth.

## App / website
- Chat mode + Code mode toggle. Asking for code inside chat still routes to a code model.
- Chat so you can refine and add to a build, not "ask once and pray".
- Image generation (separate diffusion model) for real textures instead of placeholders; the coder writes the
  texture prompts. Video later.
- A coordinator that eventually splits one request (e.g. "copper hammer, 3D, fully textured") across the
  coding model(s) and the image model.

## Screen / computer control (later, own phase)
- Thin app on the PC: it can see the screen and control apps, but the MODEL stays on a server - nothing of the
  model lives on the PC. The app streams screen + runs the actions the server decides.
- Needs a real always-on server (not GitHub Actions batch jobs). Separate hosting, decided when we get there.
- Privacy reality: screen data (which may show passwords/DMs) has to travel to the server for it to act.
- Safe scope first: local actions with approval (e.g. "open Discord and DM Bob about the mod").
- Auto-login to accounts / auto-posting (YouTube etc.): LAST, and opt-in - platforms forbid automated login and
  can ban accounts; autonomous "make videos" is beyond reliable quality today.

## Restrictions
- After training, lean on a request-side check (does the ask hint at something in restrictions.txt?) and trim the
  code scanner (tools/restrictions.py). Keep a minimal scanner for clear-malware patterns.

## Security specialist model (a main product direction)
Goal: a coding/security model companies can buy to test and protect their OWN sites/apps, respond to breaches,
and clean up their OWN exposed data. A legitimate pentest/security-tooling product.
- Knows cybersecurity deeply: how attacks work (DDoS, injection, XSS, auth flaws, phishing, doxxing exposure),
  so it can explain, test, harden and defend.
- Posture: DEFENDER + AUTHORIZED TESTER only. It never builds or runs attacks against targets the user doesn't own.
- Authorization gate: before any live action against a target, the product confirms the user owns it / has written
  permission (a "scope agreement", like every real pentest tool). "Erase data" = the user's OWN data or data they
  are authorized to remove - never someone else's systems.
- This keeps it powerful and legally sellable: the knowledge is the value; permission is what makes use lawful.
