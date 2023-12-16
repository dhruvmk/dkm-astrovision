import os
import zipfile

from huggingface_hub import hf_hub_download

# Zip files of segments to extract
zipfiles = {
    "clusters/train/dawn_vesta/00000000.zip": "00000000",
    "clusters/train/dawn_vesta/00000001.zip": "00000001",
    "clusters/train/dawn_vesta/00000002.zip": "00000002",
    "clusters/train/dawn_vesta/00000003.zip": "00000003"
}

for file in zipfiles:
    hf_hub_download(
        repo_id="travisdriver/astrovision-data",
        filename=file,
        repo_type="dataset",
        local_dir="data",
        local_dir_use_symlinks=False
    )

    with zipfile.ZipFile("data/" + file, "r") as zip_ref:
        zip_ref.extractall("data/" + zipfiles[file])
