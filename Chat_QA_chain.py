from prompts.prompt_factory import PromptFactory
from model_to_llm import get_llm
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from retrievers.vector_retriever import VectorRetriever
from create_db import create_db
from langchain.chains import LLMChain


class Chat_QA_chain():
    def __init__(self, file_path):
        self.file_path = file_path
        self.vector_store = create_db(file_path)
        self.retriever = VectorRetriever(self.vector_store).get_retriever(
            score_threshold=0.4, k=3)

    def build_qa_chain(self):
        """创建基于检索的问答链"""

        # 1. 获取LLM模型
        llm = get_llm()

        # 2. 获取Prompt模板
        initial_prompt = PromptFactory.get_prompt(
            chain_type="refine", prompt_name="initial")
        refine_prompt = PromptFactory.get_prompt(
            chain_type="refine", prompt_name="refine")
        document_prompt = PromptFactory.get_prompt(
            chain_type="refine", prompt_name="document_prompt")
        # 封装 LLMChain
        initial_llm_chain = LLMChain(llm=llm, prompt=initial_prompt)
        refine_llm_chain = LLMChain(llm=llm, prompt=refine_prompt)

        # 3. 创建问答链
        refine_chain = RefineDocumentsChain(
            initial_llm_chain=initial_llm_chain,
            refine_llm_chain=refine_llm_chain,
            document_prompt=document_prompt,
            document_variable_name="context",     # 与 prompt 中的变量对应
            initial_response_name="prev_response"  # 与 refine prompt 中的变量对应
        )
        qa_chain = RetrievalQA(
            retriever=self.retriever,
            combine_documents_chain=refine_chain,
            return_source_documents=True,
        )
        print("✅ refine 模式 QA 链构建完成！")
        return qa_chain
