"""Upload the per-dataset CE models as a GitHub Release asset."""
import os
import sys

import requests

TOKEN = os.environ["GITHUB_TOKEN"]
REPO = "dyhsup/rag"
PROXY = {"http": "http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128",
         "https": "http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128"}
API = f"https://api.github.com/repos/{REPO}"
HEAD = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github+json"}
TARBALL = "/tmp/ce-per-dataset-5x.tar.gz"
SHA = "/tmp/ce5.sha256"


def main():
    tag = "models-v1"
    r = requests.get(f"{API}/releases/tags/{tag}", headers=HEAD, proxies=PROXY, timeout=60)
    if r.status_code == 200:
        rel = r.json()
    else:
        r = requests.post(f"{API}/releases", headers=HEAD, proxies=PROXY, timeout=60, json={
            "tag_name": tag,
            "name": "Per-dataset CE models (5 datasets)",
            "body": "Five cross-encoders fine-tuned per dataset (musique, wiki2, hotpot, squad, trivia).\n"
                    "Trained from cross-encoder/ms-marco-MiniLM-L-6-v2; see README-models.md in the asset.",
        })
        r.raise_for_status()
        rel = r.json()
    print("release", rel["id"], rel["html_url"], flush=True)

    existing = {a["name"]: a for a in rel.get("assets", [])}
    for path, name in [(TARBALL, "ce-per-dataset-5x.tar.gz"), (SHA, "ce-per-dataset-5x.tar.gz.sha256")]:
        if name in existing:
            print("exists", name, existing[name]["browser_download_url"], flush=True)
            continue
        size = os.path.getsize(path)
        print(f"uploading {name} ({size/1e6:.0f} MB)", flush=True)
        with open(path, "rb") as f:
            r = requests.post(
                f"https://uploads.github.com/repos/{REPO}/releases/{rel['id']}/assets",
                params={"name": name}, headers={**HEAD, "Content-Type": "application/octet-stream"},
                data=f, proxies=PROXY, timeout=(60, 7200))
        print(r.status_code, r.text[:200], flush=True)
        r.raise_for_status()
        print("uploaded", r.json().get("browser_download_url"), flush=True)


if __name__ == "__main__":
    main()
