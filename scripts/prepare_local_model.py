"""Download and verify the same local reranker used by relational_runtime."""
from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

if __name__ == "__main__":
    snapshot_download(
        repo_id=MODEL,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "pytorch_model.bin"],
    )
    model = CrossEncoder(MODEL, device="cpu", max_length=128, local_files_only=True)
    scores = model.predict([("MAML convergence theory", "Theoretical convergence of MAML")])
    print(f"Local reranker ready: {MODEL}; pairs checked: {len(scores)}")
