import os

import kaggle


def download_data():
    kaggle.api.authenticate()

    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)

    kaggle.api.dataset_download_files(
        "splcher/animefacedataset",
        path=output_dir,
        unzip=True
    )