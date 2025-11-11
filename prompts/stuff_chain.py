from langchain_core.prompts import PromptTemplate


class StuffChainPrompts:
    # 定义信道建模领域的专业技术助手的Prompt模板-------Stuff Chain
    @staticmethod
    def get_prompt():
        template = """你是一名信道建模领域的专业技术助手，请你基于以下提供的文档内容，简明扼要地回答用户的问题。
            要求：
            1. 只使用提供的文档内容回答问题，禁止使用外部知识。
            2。 如果文档内容不足以回答问题，直接说明“文档中未找到答案”。
            3. 引用具体的文档片段作为依据，确保回答准确且有据可依。
            4. 回答需专业、严谨、逻辑清晰、连贯自然。

            文档内容：
            {context}

            用户问题：{question}

            请给出简明扼要的回答：
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        return prompt
