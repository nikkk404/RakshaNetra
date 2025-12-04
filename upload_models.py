from huggingface_hub import HfApi, upload_folder
from decouple import config

api = HfApi()
repo = config("HF_MODEL_REPO")

files = [
    ("models/model.safetensors", "model.safetensors"),
    ("models/urgency_model.safetensors", "urgency_model.safetensors"),
    ("models/config.json", "config.json"),
    ("models/label_encoder.pkl", "label_encoder.pkl"),
    ("models/urgency_encoder.pkl", "urgency_encoder.pkl"),
]

# Upload individual files
for local_path, remote_path in files:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_path,
        repo_id=repo
    )
    print(f"Uploaded: {remote_path}")

# Upload tokenizer folder
upload_folder(
    repo_id=repo,
    folder_path="models/bert_tokenizer",
    path_in_repo="bert_tokenizer"
)
print("Uploaded: bert_tokenizer folder")
