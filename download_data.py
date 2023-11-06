import os
import zipfile

from huggingface_hub import hf_hub_download

# Zip files of segments to extract
zipfiles = {
    "segments/cas_epim/opus.zip": "cas_epim",
    "segments/cas_mimas/sbmt.zip": "cas_mimas",
    "segments/dawn_ceres/2015293_c6_orbit125.zip": "dawn_ceres",
    "segments/dawn_vesta/2011205_rc3.zip": "dawn_vesta",
    "segments/haya_itokawa/20050909_20051119.zip": "haya_itokawa",
    "segments/rosetta_lutetia/sbmt.zip": "rosetta_lutetia"
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