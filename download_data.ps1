$ErrorActionPreference = "Stop"

# Array of URLs for the DDR packs
$URLS = @(
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1509", # A3
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1293", # A20 PLUS
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1292", # A20
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1148", # A
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1709", # WORLD
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=802",  # X3 VS 2ndMIX
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=546",  # X2
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=295",  # X
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=864",  # 2014
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=845",  # 2013
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=77",   # SuperNOVA2
  "https://zenius-i-vanisher.com/v5.2/download.php?type=ddrpack&categoryid=1"     # SuperNOVA
)

# Directory to save and extract files
$DOWNLOAD_DIR = "data/raw"
$EXTRACT_DIR = "$DOWNLOAD_DIR/ddr_official"

# Create directories if they don't exist
if (-not (Test-Path -Path $EXTRACT_DIR)) {
    New-Item -ItemType Directory -Force -Path $EXTRACT_DIR | Out-Null
}

# Loop through the URLs and download/extract
for ($i = 0; $i -lt $URLS.Count; $i++) {
    $url = $URLS[$i]
    $pack_num = $i + 1
    
    Write-Host "--- Downloading Pack $pack_num of $($URLS.Count) ---"
    
    $temp_download_path = Join-Path $DOWNLOAD_DIR "temp_download.zip"
    
    try {
        # Download the file
        Invoke-WebRequest -Uri $url -OutFile $temp_download_path -UserAgent "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
        
        if (Test-Path $temp_download_path) {
            Write-Host "--- Extracting $temp_download_path ---"
            # Extract the zip file
            Expand-Archive -Path $temp_download_path -DestinationPath $EXTRACT_DIR -Force
            
            # Clean up the zip file
            Remove-Item -Path $temp_download_path -Force
        } else {
            Write-Error "Error: Could not download file for URL: $url"
            exit 1
        }
    } catch {
        Write-Error "Failed to download or extract: $_"
        exit 1
    }
    
    Write-Host "--- Pack $pack_num completed ---"
}

Write-Host "--- All packs downloaded and extracted successfully! ---"
