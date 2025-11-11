from langchain_core.prompts import PromptTemplate
from model_to_llm import get_llm


class RefineChainPrompts:
    # 定义信道建模领域的专业技术助手的Prompt模板-------Refine Chain
    @staticmethod
    def get_initial_prompt():
        # 定义初始 prompt（用于第一个文档）
        initial_template = """你是一名信道建模领域的专业技术助手，下面是文档片段和用户问题，请根据文档内容给出初步回答，请用纯文本回答。

        文档内容：
        {context}

        用户问题：{question}

        请给出简明扼要的初步回答："""

        initial_prompt = PromptTemplate(
            template=initial_template,
            input_variables=["context", "question"]
        )
        return initial_prompt

    @staticmethod
    def get_refine_prompt():
        # 定义精炼 prompt（用于后续文档迭代精炼）
        refine_template = """你是一名信道建模领域的专业技术助手，我们将根据新的文档内容，在保持原回答逻辑严谨的前提下，改进之前给出的回答。

        已有回答：{prev_response}

        现在有新的文档内容需要参考：
        {context}

        请基于新文档内容完善已有回答：
        - 如果新内容与已有回答一致，保持原回答
        - 如果新内容提供了更多信息，补充到回答中
        - 如果新内容与已有回答矛盾，以新内容为准

        完善后的回答："""

        refine_prompt = PromptTemplate(
            template=refine_template,
            input_variables=["prev_response", "context", "question"]
        )
        return refine_prompt

    @staticmethod
    def get_document_prompt():
        llm = get_llm()

        # 文档格式化模板
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}"
        )
        return document_prompt
