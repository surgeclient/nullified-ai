"""Upload a folder to a private Hugging Face dataset repo under the token owner's account.

Usage (HF_TOKEN must be set in the environment):
    python tools/upload_hf.py --folder reference --repo nullified-ai-reference --path-in-repo 1.21.11
"""
import argparse
import os

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--repo", required=True, help="repo name (the account is taken from the token)")
    parser.add_argument("--path-in-repo", default="")
    parser.add_argument("--type", default="dataset", choices=["dataset", "model"])
    args = parser.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    repo_id = f"{api.whoami()['name']}/{args.repo}"
    api.create_repo(repo_id, repo_type=args.type, private=True, exist_ok=True)
    api.upload_folder(folder_path=args.folder, repo_id=repo_id, repo_type=args.type,
                      path_in_repo=args.path_in_repo, commit_message=f"Update {args.path_in_repo or 'data'}")
    print(f"uploaded {args.folder} -> https://huggingface.co/datasets/{repo_id}" if args.type == "dataset"
          else f"uploaded {args.folder} -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
