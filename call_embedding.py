from langchain_huggingface import HuggingFaceEmbeddings

# 获取嵌入模型
def get_embedding(embedding_model):
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings