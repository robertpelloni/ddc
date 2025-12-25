import requests
import re
import sys

BASE_URL = "https://zenius-i-vanisher.com/v5.2/viewsimfilecategory.php?categoryid={}"
DOWNLOAD_PATTERN = re.compile(r'Download Pack', re.IGNORECASE)

def find_download_links(start_id, end_id):
    """
    Iterates through a range of category IDs, fetches the corresponding pages,
    and checks for a "Download Pack" link.
    """
    found_links = []
    for category_id in range(start_id, end_id + 1):
        url = BASE_URL.format(category_id)
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status() # Raise an exception for bad status codes

            if DOWNLOAD_PATTERN.search(response.text):
                print(f"Found potential download page: {url}")
                found_links.append(url)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}", file=sys.stderr)
            continue
    
    return found_links

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python find_ddr_packs.py <start_id> <end_id>")
        sys.exit(1)
        
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    
    links = find_download_links(start, end)
    
    if links:
        print("\n--- Found Download Pages ---")
        for link in links:
            print(link)
    else:
        print("\nNo download pages found in the specified range.")
