import os
from huggingface_hub import hf_hub_download
from decouple import config


REPO = config("HF_MODEL_REPO")
CACHE_DIR = "models_cache"


def load_main_model():
    model_path = hf_hub_download(
        repo_id=REPO,
        filename="model.safetensors",
        cache_dir=CACHE_DIR,
    )
    return model_path


def load_urgency_model():
    model_path = hf_hub_download(
        repo_id=REPO,
        filename="urgency_model.safetensors",
        cache_dir=CACHE_DIR,
    )
    return model_path


def load_tokenizer():
    tokenizer_path = hf_hub_download(
        repo_id=REPO,
        filename="bert_tokenizer/vocab.txt",   # HF bundles folder contents automatically
        cache_dir=CACHE_DIR,
    )
    # Return folder, not file
    return os.path.dirname(tokenizer_path)


def load_encoders():
    label_enc_path = hf_hub_download(
        repo_id=REPO,
        filename="label_encoder.pkl",
        cache_dir=CACHE_DIR,
    )
    urgency_enc_path = hf_hub_download(
        repo_id=REPO,
        filename="urgency_encoder.pkl",
        cache_dir=CACHE_DIR,
    )

    return label_enc_path, urgency_enc_path
