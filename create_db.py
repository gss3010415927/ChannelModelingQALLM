import os
import re
from call_embedding import get_embedding
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,      # 网页
    PyPDFLoader,        # PDF
    UnstructuredWordDocumentLoader,  # Word 文档
    TextLoader,         # 文本文件
    DirectoryLoader,    # 目录
    CSVLoader,          # CSV
)

# 创建或加载向量数据库
def create_db(data_path, embeddings_model="BAAI/bge-large-zh-v1.5", chunk_size=1000):
    """ 
    该函数用于加载 PDF 文件，切分文档，生成文档的嵌入向量，创建向量数据库。
    参数：
    - data_path: 可以是单个文件路径或目录路径
    - embeddings_model: 使用的嵌入模型名称，默认为 "BAAI/bge-large-zh-v1.5"
    """

    persist_dir = "./chroma_db_all_docs"   # ✅ 多文档统一存入同一个库

    # 如果向量库存在，直接加载
    if os.path.exists(persist_dir):
        print(f"🔄 发现已存在向量库，直接加载: {persist_dir}")
        embeddings = get_embedding(embeddings_model)
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    print("📥 首次运行，开始构建向量库...")

    # 1. 加载数据
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
    
    print(f"📄 文档载入完成，共 {len(docs)} 条记录")

    # 2. 数据清洗
    for doc in docs:
        # 匹配任何字符之间的换行符（包括中文、英文、数字等）
        pattern = re.compile(r'(.)\n(.)', re.DOTALL)
        doc.page_content = re.sub(pattern, r'\1\2', doc.page_content)
    print("文本清洗完成")

    # 3. 文本切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.1),
        separators=["\n\n", "\n", " ", ""]  # 优先按段落分割
    )
    splits = text_splitter.split_documents(docs)

    # 4. 写入向量数据库
    embeddings = get_embedding(embeddings_model)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print("✅ 向量库构建完成！已持久化存储。")
    return vectorstore
