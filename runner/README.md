# nullified-runner (public repo)

This folder is the content of the **public** `nullified-runner` repo. On GitHub's free servers (public repos
get 4 CPUs, 16 GB RAM and unlimited minutes) the `build` workflow:

1. checks out the **private** `nullified-ai` code (with the `NULLIFIED_TOKEN` secret),
2. downloads the model (gpt-oss-20b by default, or your own trained one) and runs it on the CPU with llama.cpp,
3. writes the mod, compiles it, boots a real 1.21.11 server with it, fixes problems, and packages the `.jar` + `.zip`.

Your PC only starts the job and downloads the result.

## Set it up (once)

1. **Create the repo:** on https://github.com/new, name it `nullified-runner`, choose **Public**, and click **Create repository**.
2. **Add the workflow:** in the new repo click **Add file → Create new file**. Name it `.github/workflows/build.yml`,
   paste in the contents of `runner/.github/workflows/build.yml` from nullified-ai, and click **Commit changes**.
3. **Make a token that lets the runner read the private code:** on
   https://github.com/settings/personal-access-tokens/new, pick **Only select repositories → nullified-ai**, set
   **Contents → Read-only**, then generate and copy it.
4. **Add the secrets:** in nullified-runner go to **Settings → Secrets and variables → Actions → New repository secret**:
   - `NULLIFIED_TOKEN`: the token from step 3
   - `HF_TOKEN`: your Hugging Face token (read access). This is needed for the 1.21.11 reference docs and later
     for your own private model.
5. **Update your PC's token:** your `GITHUB_TOKEN` must also cover `nullified-runner` with
   **Actions: Read and write**. You can edit your existing token and add the repo.

## Build a mod

- **From the website:** open nullified-runner → **Actions → build → Run workflow**, type the mod you want and
  click **Run**. When the run finishes (about 45–90 min), download **mod** under **Artifacts** at the bottom of the run page.
- **From your PC:** `python local/nullified_remote.py --request "a ruby sword that sets mobs on fire"`
  (downloads into `output/`).

## Good to know

- **Anyone can see what you build.** Because the repo is public, anyone can see the requests, the logs and the
  built mods (artifacts). The code, `restrictions.txt`, the model and the tokens are **not** visible.
- **It's slow.** The model runs on a CPU, so one attempt at a mod takes about 15–25 minutes. Several builds can
  run at the same time.
- **To use your own trained model later:** pass `--model-repo <you>/<repo> --model-file <file>.gguf`, or change
  the defaults in `build.yml`.
