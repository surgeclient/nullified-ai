"""Build a mod entirely on GitHub: this script only starts the job and downloads the result.

The model, the compiling, the Minecraft test server and the packaging all run on GitHub's free servers
(.github/workflows/nullified.yml in this repo, which must be public). Your PC needs nothing but Python and a
GitHub token.

    python local/nullified_remote.py --request "a ruby sword that sets mobs on fire"

Env: GITHUB_TOKEN (fine-grained, this repo: Actions read+write, Contents read),
     NULLIFIED_REPO (default: this checkout's GitHub repo).
Output: output/<modid>/<modid>-1.0.0.jar and <modid>-project.zip, plus output/result.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from github_checks import GitHubActions, github_repo  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--request", required=True, help="the mod you want")
    p.add_argument("--fix-rounds", type=int, default=3)
    p.add_argument("--repo", default="", help="owner/repo on GitHub (default: this checkout's origin)")
    p.add_argument("--model-repo", default=os.environ.get("NULLIFIED_MODEL_REPO", ""),
                   help="Hugging Face repo of your trained model (default: gpt-oss-20b)")
    p.add_argument("--model-file", default=os.environ.get("NULLIFIED_MODEL_FILE", ""))
    p.add_argument("--ref", default="", help="branch to run (default: the repo's default branch)")
    args = p.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        sys.exit("Set GITHUB_TOKEN (see local/README.md) and open a new terminal.")
    repo = args.repo or github_repo()
    inputs = {"request": args.request, "fix_rounds": str(args.fix_rounds)}
    if args.model_repo:
        inputs["model_repo"] = args.model_repo
    if args.model_file:
        inputs["model_file"] = args.model_file

    started = time.time()
    print(f"Starting the build on GitHub ({repo}). This usually takes 45-90 minutes on the free servers;")
    print("you can close this window - the job keeps running and the .jar stays in the run's Artifacts for 7 days.")
    gh = GitHubActions(repo, token, args.ref)
    run, files = gh.dispatch_and_wait("nullified.yml", inputs, artifact="mod", poll=30,
                                      on_start=lambda r: print(f"Watch it live: {r['html_url']}", flush=True))

    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    for name, data in files.items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(data)
    if "result.json" not in files:
        sys.exit(f"The build ended '{run['conclusion']}' without a result. See the log: {run['html_url']}")

    r = json.loads(files["result.json"])
    if r["refused"]:
        print(f"Refused: {r['refused']}")
    elif r["ok"]:
        jars = [n for n in files if n.endswith(".jar")]
        print(f"PLAN:\n{r['plan']}\n\nDone in {(time.time() - started) / 60:.0f} min:")
        for n in files:
            if n.endswith((".jar", ".zip")):
                print(f"  {(out / n).resolve()}")
        if jars:
            print("Put the .jar in %appdata%\\.minecraft\\mods (needs Fabric Loader + Fabric API for 1.21.11).")
    else:
        print(f"Could not make a mod that passes every check ({' -> '.join(map(str, r['stage_history']))}). "
              "Last problems:")
        for e in r["errors"][:6]:
            print("  " + (e.splitlines() or [""])[0][:200])
        print(f"Full log: {run['html_url']}")


if __name__ == "__main__":
    main()
