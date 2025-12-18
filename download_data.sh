#!/bin/bash

set -e

# Array of URLs for the DDR packs
URLS=(
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1509" # A3
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1293" # A20 PLUS
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1292" # A20
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1148" # A
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1709" # WORLD
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=802"  # X3 VS 2ndMIX
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=546"  # X2
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=295"  # X
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=864"  # 2014
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=845"  # 2013
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=77"   # SuperNOVA2
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1"    # SuperNOVA
)

# Directory to save and extract files
DOWNLOAD_DIR="data/raw"
EXTRACT_DIR="$DOWNLOAD_DIR/ddr_official"
mkdir -p "$EXTRACT_DIR"

# Loop through the URLs and download/extract
for i in "${!URLS[@]}"; do
  url="${URLS[$i]}"
  pack_num=$((i+1))
  
  echo "--- Downloading Pack $pack_num of ${#URLS[@]} ---"
  
  # Download the file and capture the real filename from the headers
  temp_download_path="$DOWNLOAD_DIR/temp_download.zip"
  wget --user-agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36" \
       --content-disposition \
       -O "$temp_download_path" \
       "$url"
       
  if [ -f "$temp_download_path" ]; then
    echo "--- Extracting $temp_download_path ---"
    unzip -o "$temp_download_path" -d "$EXTRACT_DIR"
    rm "$temp_download_path" # Clean up the zip file after extraction
  else
    echo "Error: Could not download file for URL: $url"
    exit 1
  fi
  
  echo "--- Pack $pack_num completed ---"
done

echo "--- All packs downloaded and extracted successfully! ---"
