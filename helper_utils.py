import os
import zipfile
import json


def download_dataset(
    dataset: str = "grassknoted/asl-alphabet",
    download_dir: str = "./data",
    kaggle_username: str = None,
    kaggle_key: str = None,
):
    """
    Download an ASL dataset from Kaggle for computer vision training.

    Args:
        dataset:         Kaggle dataset slug in the format 'owner/dataset-name'.
                         Default: 'grassknoted/asl-alphabet' (87,000 images, 29 classes).
                         Other options:
                           - 'debashishsau/aslpng'
                           - 'ayuraj/asl-dataset'
        download_dir:    Local directory to save and extract the dataset.
        kaggle_username: Your Kaggle username. If None, reads from ~/.kaggle/kaggle.json.
        kaggle_key:      Your Kaggle API key.  If None, reads from ~/.kaggle/kaggle.json.
    """
    # --- 1. Set credentials ---
    if kaggle_username and kaggle_key:
        os.environ["KAGGLE_USERNAME"] = kaggle_username
        os.environ["KAGGLE_KEY"] = kaggle_key
    else:
        _load_credentials_from_file()

    # --- 2. Skip download if data already exists ---
    dataset_name = dataset.split("/")[-1].replace("-", "_")
    expected_dirs = [
        os.path.join(download_dir, d)
        for d in (f"{dataset_name}_train", f"{dataset_name}_test")
    ]
    if any(os.path.isdir(d) for d in expected_dirs):
        print(f"Dataset already exists at '{os.path.abspath(download_dir)}'. Skipping download.")
        return download_dir

    # --- 3. Import kaggle AFTER credentials are set ---
    try:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApiExtended as KaggleApi
        except ImportError:
            from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError("kaggle package not found. Install it with: pip install kaggle")

    api = KaggleApi()
    api.authenticate()

    # --- 4. Download dataset (not competition) ---
    os.makedirs(download_dir, exist_ok=True)
    print(f"Downloading dataset '{dataset}' to '{download_dir}' ...")
    api.dataset_download_files(dataset, path=download_dir, unzip=False, quiet=False)

    # --- 5. Unzip ---
    zip_name = dataset.split("/")[-1]
    zip_path = os.path.join(download_dir, f"{zip_name}.zip")
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(download_dir)
        os.remove(zip_path)
        print("Extraction complete. Zip file removed.")
    else:
        print("No zip file found — files may have been downloaded individually.")

    print(f"Dataset ready at: {os.path.abspath(download_dir)}")
    return download_dir


def _load_credentials_from_file():
    """Load Kaggle credentials from ~/.kaggle/kaggle.json if env vars are not set."""
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        raise FileNotFoundError(
            "No Kaggle credentials found.\n"
            "Either pass kaggle_username and kaggle_key to download_dataset(), "
            "or place your kaggle.json at ~/.kaggle/kaggle.json.\n"
            "Get your token at: https://www.kaggle.com/settings (Account → API → Create New Token)"
        )
    with open(kaggle_json) as f:
        creds = json.load(f)
    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]

