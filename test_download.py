import os
import requests
import zipfile
import sys

URLS = [
    "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1509",  # A3
]

DOWNLOAD_DIR = "data/raw"
EXTRACT_DIR = os.path.join(DOWNLOAD_DIR, "ddr_official")

if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR)


def download_url(url, dest_path):
    print(f"Downloading {url}...")
    sys.stdout.flush()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        r = requests.get(url, stream=True, headers=headers, timeout=30)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to {dest_path}")
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        sys.stdout.flush()
        return False


def main():
    print(f"Starting single download test...")
    sys.stdout.flush()
    url = URLS[0]
    temp_zip = os.path.join(DOWNLOAD_DIR, "temp_download_test.zip")

    if download_url(url, temp_zip):
        print("Download successful.")
    else:
        print("Download failed.")


if __name__ == "__main__":
    main()
