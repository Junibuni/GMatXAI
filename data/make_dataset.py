import os
import requests

output_dir = os.path.join(os.getcwd(), "data", "megnet")
os.makedirs(output_dir, exist_ok=True)

files = {
    "https://figshare.com/ndownloader/files/40258705": "bulk_megnet_train.pkl",
    "https://figshare.com/ndownloader/files/40258675": "bulk_megnet_val.pkl",
    "https://figshare.com/ndownloader/files/40258666": "bulk_megnet_test.pkl",
}

for url, filename in files.items():
    target_path = os.path.join(output_dir, filename)
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded and saved: {target_path}")
        else:
            print(f"Failed to download {filename} (Status code: {response.status_code})")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")