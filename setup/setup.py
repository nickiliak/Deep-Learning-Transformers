import os
import requests
import zipfile
from tqdm import tqdm


## SETUP INSTALLATION FOR GOOGLE NQ DATASET FROM THE BEIR REPO
## https://github.com/beir-cellar/beir?tab=readme-ov-file


def download_and_process(url, output_filename, extract_to_folder):
    """
    Downloads a file with a progress bar, extracts it, and deletes the zip.
    """
    
    # --- STEP 1: DOWNLOAD ---
    print(f"1. Starting download from: {url}")
    
    # Establish connection
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1KB
    
    # Progress Bar
    progress_bar = tqdm(
        total=total_size, 
        unit='iB', 
        unit_scale=True,
        desc="Downloading"
    )
    
    # Writing file
    with open(output_filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Download failed.")
        return

    # --- STEP 2: EXTRACT (UNZIP) ---
    print(f"\n2. Extracting {output_filename}...")
    
    try:
        # Create folder if it doesn't exist
        os.makedirs(extract_to_folder, exist_ok=True)
        
        with zipfile.ZipFile(output_filename, 'r') as zip_ref:
            # extractall is the standard method to unzip everything
            zip_ref.extractall(extract_to_folder) 
        print("Extraction successful.")
        
    except zipfile.BadZipFile:
        print("Error: The downloaded file is corrupt.")
        return

    # --- STEP 3: CLEANUP (DELETE ZIP) ---
    print(f"3. Deleting temporary file: {output_filename}")
    try:
        os.remove(output_filename)
        print("Cleanup successful.")
    except OSError as e:
        print(f"Error deleting file: {e}")

    print(f"\n--- PROCESS COMPLETED ---")
    print(f"Data is ready in folder: {extract_to_folder}")

if __name__ == "__main__":
    # Configuration
    dataset_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
    zip_name = "nq.zip"
    output_folder = "data" # Where the data will be extracted
    
    download_and_process(dataset_url, zip_name, output_folder)