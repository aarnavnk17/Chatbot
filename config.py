# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "./db"

COLLECTION_NAME = "knowledge_base"

CLIP_MODEL_NAME = "laion/CLIP-ViT-B32-128-multistep"
CLIP_PROCESSOR_NAME = "laion/CLIP-ViT-B32-128-multistep"
CLIP_TOKENIZER_NAME = "laion/CLIP-ViT-B32-128-multistep"

CHROMADB_COLLECTION = {
    'name': COLLECTION_NAME,
    'vector_size': 128
}
