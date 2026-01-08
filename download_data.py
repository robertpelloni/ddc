import os
import subprocess
import zipfile
import time

URLS = [
    # "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1509",  # A3 (Done)
    "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1293",  # A20 PLUS
    "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1292",  # A20
    "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1148",  # A
    "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1709",  # WORLD
    "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=802",  # X3 VS 2ndMIX
    "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=546",  # X2
    "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=295",  # X
    "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=864",  # 2014
    "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=845",  # 2013
    "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=77",  # SuperNOVA2
    "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1",  # SuperNOVA
]

# Map category ID (from URL) to a folder that is known to exist if the pack is installed
# This helps us skip packs we already have.
PACK_MARKERS = {
    "1293": "HyperTwist",  # A20 PLUS
    # "1509": "MEGALOVANIA", # A3 (Already commented out in URLS)
}

# Add logging
import sys


def log(msg):
    print(msg)
    sys.stdout.flush()


import requests

def download_url_curl(url, dest_path):
    log(f"Downloading {url} to {dest_path} using requests...")
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    headers = {"User-Agent": user_agent}

    # Check for resume
    mode = "wb"
    if os.path.exists(dest_path):
        existing_size = os.path.getsize(dest_path)
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"
            log(f"Resuming from byte {existing_size}...")

    max_retries = 10
    for attempt in range(max_retries):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                # 416 Range Not Satisfiable means we already have the whole file (probably)
                if r.status_code == 416: 
                    log("File already fully downloaded (server returned 416).")
                    return True
                
                r.raise_for_status()
                with open(dest_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Downloaded to {dest_path}")
            return True
        except Exception as e:
            print(f"Error downloading {url} (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                # If we retry, we need to update the Range header for the next attempt
                if os.path.exists(dest_path):
                    headers["Range"] = f"bytes={os.path.getsize(dest_path)}-"
                    mode = "ab"
            else:
                print("Max retries reached. Download failed.")
                return False


DOWNLOAD_DIR = "data/raw"
EXTRACT_DIR = os.path.join(DOWNLOAD_DIR, "ddr_official")
if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR)


def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False


def main():
    log(f"Starting download of {len(URLS)} packs...")
    for i, url in enumerate(URLS):
        log(f"--- Pack {i + 1}/{len(URLS)} ---")

        # Extract category ID from URL
        import urllib.parse

        parsed_url = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        cat_id = query_params.get("categoryid", [""])[0]

        log(f"Checking pack ID: {cat_id}")

        # Check if we should skip
        if cat_id in PACK_MARKERS:
            marker_folder = os.path.join(EXTRACT_DIR, PACK_MARKERS[cat_id])
            if os.path.exists(marker_folder):
                log(f"Skipping pack {cat_id} because '{PACK_MARKERS[cat_id]}' exists.")
                continue

        temp_zip = os.path.join(DOWNLOAD_DIR, f"temp_download_{i}.zip")

        # Check if we should skip (simple check if folder exists might be tricky as we don't know the folder name inside zip)
        # For now, just download everything.

        if download_url_curl(url, temp_zip):
            if extract_zip(temp_zip, EXTRACT_DIR):
                try:
                    os.remove(temp_zip)
                except OSError as e:
                    print(f"Error deleting temp zip: {e}")
            else:
                print("Failed to extract.")
                # If extraction fails, do NOT delete the zip immediately so we can inspect or retry.
                # But here we want to retry download, so maybe we SHOULD delete or rename?
                # For now, let's keep it but exit loop or continue?
                # The script continues to next pack.
        else:
            print("Failed to download.")

    print("All downloads finished.")


if __name__ == "__main__":
    main()
