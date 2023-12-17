import os
import zipfile

from huggingface_hub import hf_hub_download

# Zip files of segments to extract
zipfiles = {}
for i in range(4):
    number = str(i).zfill(8)
    zipfiles["clusters/train/dawn_vesta/" + number + ".zip"] = number

print(zipfiles)

for file in zipfiles:
    hf_hub_download(
        repo_id="travisdriver/astrovision-data",
        filename=file,
        repo_type="dataset",
        local_dir="data",
        local_dir_use_symlinks=False
    )

    if file == "00000000" or file == "00000001":
        with zipfile.ZipFile("data/" + file, "r") as zip_ref:
            zip_ref.extractall("data/")
    else:
        with zipfile.ZipFile("data/" + file, "r") as zip_ref:
            zip_ref.extractall("data/" + file)
