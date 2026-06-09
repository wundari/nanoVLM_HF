import os

os.environ["HTTP_PROXY"] = "http://proxy.nict.go.jp:3128"
os.environ["HTTPS_PROXY"] = "http://proxy.nict.go.jp:3128"

from huggingface_hub import list_repo_files, hf_hub_download, snapshot_download

repo_id = "HuggingFaceM4/FineVision"
local_dir = "../Dataset/HuggingFace/FineVision"
dataset_name = "a_okvqa"

snapshot_download(
    repo_id="HuggingFaceM4/FineVision",
    repo_type="dataset",
    allow_patterns=[f"{dataset_name}/*"],
    local_dir=local_dir,
    max_workers=16,  # parallel file downloads
    ignore_patterns=["*.md", "*.json"],  # skip metadata if you only want parquet
)
