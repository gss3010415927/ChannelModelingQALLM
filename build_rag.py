from langchain_community.document_loaders import (
    WebBaseLoader,      # 网页
    PyPDFLoader,        # PDF
    UnstructuredWordDocumentLoader,  # Word 文档
    TextLoader,         # 文本文件
    DirectoryLoader,    # 目录
    CSVLoader,          # CSV
)
import os
import re
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain

def create_vectorstore(data_path: str, chunk_size: int = 1000):
    """
    data_path 可以是：
        1) 单个文件路径: ./data/1.docx
        2) 文件目录路径: ./data/     （里面多个word/pdf/txt）
    """

    persist_dir = "./chroma_db_all_docs"   # ✅ 多文档统一存入同一个库

    # ✅ 如果向量库存在，直接加载
    if os.path.exists(persist_dir):
        print(f"🔄 发现已存在向量库，直接加载: {persist_dir}")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    print("📥 首次运行，开始构建向量库...")

    # ✅ 判断路径是文件还是目录
    docs = []
    if os.path.isdir(data_path):
        loader = DirectoryLoader(
            data_path,
            glob="**/*.pdf",  # 可以改为 "*.pdf" 或 "*.txt" 或组合
            loader_cls=PyPDFLoader
        )
        docs = loader.load()
    else:
        loader = PyPDFLoader(data_path)
        docs = loader.load()
    #------------数据清洗---------
    # 对每个文档进行文本清理
    for doc in docs:
        # 匹配任何字符之间的换行符（包括中文、英文、数字等）
        pattern = re.compile(r'(.)\n(.)', re.DOTALL)
        doc.page_content = re.sub(pattern, r'\1\2', doc.page_content)

    # pdf_page = docs[1]
    # print(f"每一个元素的类型：{type(pdf_page)}.", 
    #     f"该文档的描述性数据：{pdf_page.metadata}", 
    #     f"查看该文档的内容:\n{pdf_page.page_content}", 
    #     sep="\n------\n")
    
    print(f"📄 文档载入完成，共 {len(docs)} 条记录")

    # ✅ 分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.1),
        separators=["\n\n", "\n", " ", ""]  # 优先按段落分割
    )
    splits = text_splitter.split_documents(docs)

    # ✅ 创建嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ✅ 写入向量数据库
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print("✅ 向量库构建完成！已持久化存储。")
    return vectorstore

def evaluate_retrieval(retriever, test_cases):
    """评估检索器性能"""

    metrics = {
        "precision": [],  # 精确率
        "recall": [],     # 召回率
    }

    for query, expected_doc_ids in test_cases:
        # 执行检索
        retrieved_docs = retriever.invoke(query)
        retrieved_ids = [doc.metadata['id'] for doc in retrieved_docs]

        # 计算指标
        relevant_retrieved = set(retrieved_ids) & set(expected_doc_ids)

        precision = len(relevant_retrieved) / \
            len(retrieved_ids) if retrieved_ids else 0
        recall = len(relevant_retrieved) / \
            len(expected_doc_ids) if expected_doc_ids else 0

        metrics["precision"].append(precision)
        metrics["recall"].append(recall)

    return {
        "avg_precision": sum(metrics["precision"]) / len(metrics["precision"]),
        "avg_recall": sum(metrics["recall"]) / len(metrics["recall"])
    }

def setup_qa_chain(vectorstore):
    """创建基于检索的问答链"""

    # 创建检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.1,  # 只返回相似度>0.2的结果
            "k": 10
        }
    )

    # 创建 Ollama LLM 实例
    ollama_llm = ChatOllama(
        model="qwen3:8b",
        temperature=0.1,
        max_tokens=1024,
        top_p=0.7,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["<|endoftext|>"]
    )

    # # 定义自定义prompt模板（可选）
    # template = """你是一名信道建模领域的专业技术助手，请你基于以下提供的文档内容，简明扼要地回答用户的问题。
    # 要求：
    # 1. 只使用提供的文档内容回答问题，禁止使用外部知识。
    # 2。 如果文档内容不足以回答问题，直接说明“文档中未找到答案”。
    # 3. 引用具体的文档片段作为依据，确保回答准确且有据可依。
    # 4. 回答需专业、严谨、逻辑清晰、连贯自然。

    # 文档内容：
    # {context}

    # 用户问题：{question}

    # 请给出简明扼要的回答：
    # """

    # prompt = PromptTemplate(
    #     template=template,
    #     input_variables=["context", "question"]
    # )


    # # 创建问答链
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=ollama_llm,
    #     chain_type="refine",    # 使用 refine 策略
    #     retriever=retriever,
    #     chain_type_kwargs={"prompt": prompt},
    #     return_source_documents=True
    # )

    # 定义初始 prompt（用于第一个文档）
    initial_template = """你是一名信道建模领域的专业技术助手，下面是文档片段和用户问题，请根据文档内容给出初步回答。

    文档内容：
    {context}

    用户问题：{question}

    请给出简明扼要的初步回答："""

    initial_prompt = PromptTemplate(
        template=initial_template,
        input_variables=["context", "question"]
    )

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

    # ---------------------
    # 2) 封装 LLMChain
    # ---------------------
    initial_llm_chain = LLMChain(llm=ollama_llm, prompt=initial_prompt)
    refine_llm_chain = LLMChain(llm=ollama_llm, prompt=refine_prompt)

    # ---------------------
    # 3) 文档格式化模板
    # ---------------------
    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )
    # -----------------
    # 4) 创建 RefineDocumentsChain
    # -----------------
    refine_chain = RefineDocumentsChain(
        initial_llm_chain=initial_llm_chain,
        refine_llm_chain=refine_llm_chain,
        document_prompt=document_prompt,
        document_variable_name="context",     # 与 prompt 中的变量对应
        initial_response_name="prev_response"  # 与 refine prompt 中的变量对应
    )
    # -----------------
    # 5) 创建 RetrievalQA
    # -----------------
    qa_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=refine_chain,
        return_source_documents=True,
    )
    # 创建问答链
    # qa_chain = RetrievalQA.from_chain_type(
    #     #llm=ollama_llm,
    #     chain_type="refine",
    #     retriever=retriever,
    #     chain_type_kwargs={
    #         "initial_llm_chain": initial_llm_chain,
    #         "refine_llm_chain": refine_llm_chain,
    #     },
    #     return_source_documents=True
    # )

    print("✅ refine 模式 QA 链构建完成！")

    return qa_chain

def query(question: str, qa_chain, show_sources: bool = True): 
    """执行问答，增加异常捕获"""

    try:
        # 调用 QA 链
        result = qa_chain.invoke({"query": question})
        answer = result.get('result', "文档中未找到答案")

        # 显示参考来源
        if show_sources and 'source_documents' in result:
            print("\n📚 参考来源:")
            for doc in result['source_documents']:
                print(f"- 来源: {doc.metadata.get('source', 'N/A')}")
                print(f"  内容: {doc.page_content[:200]}...\n")

    except (IndexError, KeyError, TypeError) as e:
        # 捕获 refine_chain 内部 docs 为空等异常
        answer = "文档中未找到答案"
        if show_sources:
            print("\n📚 参考来源: 无")

    return answer

if __name__ == "__main__":
    vectorstore = create_vectorstore("./data")

    qa_chain = setup_qa_chain(vectorstore)

    answer = query("什么是人体阴影？", qa_chain, show_sources=True)
    print("\n🤖 回答:")
    print(answer)