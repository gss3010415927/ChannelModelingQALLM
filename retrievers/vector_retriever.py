class VectorRetriever:
    """ 基于向量检索的类 """

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def get_retriever(self, score_threshold=0.5, k=3):
        """ 获取向量检索器 """
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": score_threshold,  # 只返回相似度>score_threshold的结果
                "k": k
            }
        )
        return retriever
