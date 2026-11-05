import os, time, requests
TOKEN=os.environ["GITHUB_TOKEN"]; REPO="dyhsup/rag"
P={"http":"http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128","https":"http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128"}
H={"Authorization":f"token {TOKEN}","Accept":"application/vnd.github+json"}
api=f"https://api.github.com/repos/{REPO}"
rel=requests.get(f"{api}/releases/tags/models-v1",headers=H,proxies=P,timeout=60).json()
rid=rel["id"]; print("release",rid,flush=True)
for a in rel.get("assets",[]):
    if a["name"]=="tiny.txt":
        requests.delete(f"{api}/releases/assets/{a['id']}",headers=H,proxies=P,timeout=30)
files=[f"/tmp/ce_{d}.tar.gz" for d in ["musique","wiki2","hotpot","squad","trivia"]]+["/tmp/ce_models.sha256"]
have={a["name"] for a in rel.get("assets",[])}
for path in files:
    name=os.path.basename(path)
    if name in have:
        print("skip",name,flush=True); continue
    for attempt in range(3):
        try:
            print(f"upload {name} {os.path.getsize(path)/1e6:.0f}MB try{attempt+1}",flush=True)
            with open(path,"rb") as f:
                r=requests.post(f"https://uploads.github.com/repos/{REPO}/releases/{rid}/assets",
                    params={"name":name},headers={**H,"Content-Type":"application/octet-stream"},
                    data=f,proxies=P,timeout=(60,3600))
            print(r.status_code,flush=True)
            if r.status_code in (200,201):
                print("ok",r.json()["browser_download_url"],flush=True); break
        except Exception as e:
            print("err",repr(e)[:200],flush=True)
        time.sleep(5)
