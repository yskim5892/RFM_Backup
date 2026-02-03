#!/usr/bin/env python3
import argparse
import json
import os
import tarfile
import time
import urllib.request
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed

# 공식 스크립트(ycb_downloader.py)에서 쓰는 규칙:
# base_url + "objects.json" 에 object name list가 있음.  :contentReference[oaicite:2]{index=2}
BASE_URL = "https://ycb-benchmarks.s3.amazonaws.com/data/"
OBJECTS_JSON_URL = BASE_URL + "objects.json"

def tgz_url(obj_name: str, file_type: str) -> str:
    if file_type in ["berkeley_rgbd", "berkeley_rgb_highres"]:
        return f"{BASE_URL}berkeley/{obj_name}/{obj_name}_{file_type}.tgz"
    elif file_type == "berkeley_processed":
        return f"{BASE_URL}berkeley/{obj_name}/{obj_name}_berkeley_meshes.tgz"
    else:
        # google_16k / google_64k / google_512k
        return f"{BASE_URL}google/{obj_name}_{file_type}.tgz"


def fetch_all_objects() -> list:
    with urllib.request.urlopen(OBJECTS_JSON_URL, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["objects"]


def download_with_retries(url: str, out_path: str, retries: int = 3) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return

    last_err = None
    for i in range(retries):
        try:
            urllib.request.urlretrieve(url, out_path)
            if os.path.getsize(out_path) == 0:
                raise RuntimeError("downloaded file is empty")
            return
        except (HTTPError, URLError, RuntimeError) as e:
            last_err = e
            # backoff
            time.sleep(1.0 * (i + 1))
    raise RuntimeError(f"Download failed after {retries} retries: {url} ({last_err})")


def extract_tgz(tgz_path: str, out_dir: str) -> None:
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(out_dir)


def already_has_textured_obj(out_dir: str, obj_name: str) -> bool:
    # 폴더 구조가 배포 버전에 따라 조금 달라서 "textured.obj"가 존재하는지로 스킵 판단
    for r, _, files in os.walk(out_dir):
        if obj_name not in r:
            continue
        if "textured.obj" in files:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./ycb", help="root folder for YCB assets")
    ap.add_argument(
        "--type",
        default="google_16k",
        choices=["google_16k", "google_64k", "google_512k", "berkeley_processed"],
        help="mesh source/type"
    )
    ap.add_argument("--all", action="store_true", help="download ALL objects listed in objects.json")
    ap.add_argument("--objects", nargs="*", default=[], help="download only these objects (e.g. 011_banana 006_mustard_bottle)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel download jobs")
    ap.add_argument("--keep_tgz", action="store_true", help="keep .tgz files after extraction")
    ap.add_argument("--skip_existing", action="store_true", help="skip objects that already have textured.obj")

    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)

    if args.all:
        objects = fetch_all_objects()
    else:
        objects = args.objects

    if not objects:
        raise SystemExit("No objects specified. Use --all or --objects 011_banana ...")

    os.makedirs(out_dir, exist_ok=True)

    # 1) download all tgz in parallel
    tasks = []
    tgz_dir = os.path.join(out_dir, "_tgz")
    os.makedirs(tgz_dir, exist_ok=True)

    for obj in objects:
        if args.skip_existing and already_has_textured_obj(out_dir, obj):
            print(f"[skip existing] {obj}")
            continue

        url = tgz_url(obj, args.type)
        tgz_name = f"{obj}_{args.type}.tgz" if args.type != "berkeley_processed" else f"{obj}_berkeley_meshes.tgz"
        tgz_path = os.path.join(tgz_dir, tgz_name)
        tasks.append((obj, url, tgz_path))

    print(f"[plan] objects={len(objects)}, to_download={len(tasks)}, type={args.type}, out_dir={out_dir}")

    # Download
    errors = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(download_with_retries, url, tgz_path): (obj, url, tgz_path) for (obj, url, tgz_path) in tasks}
        for fut in as_completed(futs):
            obj, url, tgz_path = futs[fut]
            try:
                fut.result()
                print(f"[downloaded] {obj} -> {tgz_path}")
            except Exception as e:
                print(f"[failed] {obj} ({url}) : {e}")
                errors.append((obj, url, str(e)))

    if errors:
        print("\n[warning] some downloads failed. Re-run with the same args to retry these objects.")
        for obj, url, err in errors[:20]:
            print(f"  - {obj}: {err} ({url})")

    # 2) extract sequentially (디스크/CPU 안정성)
    for obj, url, tgz_path in tasks:
        if not os.path.exists(tgz_path) or os.path.getsize(tgz_path) == 0:
            continue
        if args.skip_existing and already_has_textured_obj(out_dir, obj):
            continue

        print(f"[extract] {obj} <- {os.path.basename(tgz_path)}")
        try:
            extract_tgz(tgz_path, out_dir)
            if not args.keep_tgz:
                os.remove(tgz_path)
        except Exception as e:
            print(f"[extract failed] {obj}: {e}")

    print("\n[done] If you want to find textured.obj for a specific object:")
    print(f"  find {out_dir} -path '*011_banana*' -name textured.obj")


if __name__ == "__main__":
    main()

