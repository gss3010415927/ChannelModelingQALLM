from langchain_core.prompts import PromptTemplate
from model_to_llm import get_llm


class RefineChainPrompts:
    """Refine Chain 专用 Prompt 模板 —— 严格基于检索内容，不得臆造"""

    @staticmethod
    def get_initial_prompt():
        """
        初始 prompt：仅基于检索内容回答用户问题。
        模型必须遵守：
        - 不得引用外部知识；
        - 若文档中没有相关信息，只能回答固定句式。
        """
        initial_template = """你是一名信道建模领域的专业技术助手。
你的任务是根据给定的【检索到的文档内容】回答用户问题。

请严格遵守以下规则：
1. 仅依据提供的文档内容（即 context 部分），不得添加、推测或扩展任何外部知识；
2. 如果文档中没有相关信息，请只回答：
   “根据已检索内容无法确定”；
3. 禁止输出模型的思考过程、解释、分析或其他说明；
4. 如果文档内容为空（即未检索到任何内容），请直接回答：“根据已检索内容无法确定”；
5. **必须使用 Markdown 格式输出内容**，包括标题（#、##、###）、列表（-）、表格（|…|）、加粗（**…**）等。

---------------------
【文档内容】
{context}

【用户问题】
{question}

---------------------
请基于文档内容直接作答，且严格使用 Markdown 格式：
"""

        return PromptTemplate(
            template=initial_template,
            input_variables=["context", "question"]
        )

    @staticmethod
    def get_refine_prompt():
        """
        精炼 prompt：基于新增文档修正已有回答。
        模型仍然必须仅依据文档，不得臆造。
        """
        refine_template = """你是一名信道建模领域的专业技术助手。
我们将基于新的【检索内容】完善之前的回答。

请严格遵守以下规则：
1. 仅依据提供的文档内容（即 context 部分），不得使用外部知识或主观臆测；
2. 若新内容与已有回答一致，则保持不变；
3. 若新内容提供更多细节，请补充；
4. 若新内容与原回答矛盾，则以新内容为准；
5. 如果文档中没有相关信息，请只回答：
   “根据已检索内容无法确定”；
6. 禁止解释、举例、联想或生成与文档无关的文本；
7. **必须使用 Markdown 格式输出内容**，包括标题（#、##、###）、列表（-）、表格（|…|）、加粗（**…**）等。

---------------------
【已有回答】
{prev_response}

【新的文档内容】
{context}

【用户问题】
{question}

---------------------
请基于文档内容给出完善后的最终回答，且严格使用 Markdown 格式：
"""

        return PromptTemplate(
            template=refine_template,
            input_variables=["prev_response", "context", "question"]
        )

    @staticmethod
    def get_document_prompt():
        """文档格式化模板（用于拼接检索内容）"""
        return PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}"
        )
