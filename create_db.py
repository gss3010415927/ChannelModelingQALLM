import os
import re
from call_embedding import get_embedding
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_db(data_path, embeddings_model="BAAI/bge-large-zh-v1.5", chunk_size=1000):
    """
    加载 PDF / Word / 文本文件，切分文档并创建持久化向量数据库。
    """
    persist_dir = "./chroma_db_all_docs"

    # ✅ 若存在旧库但未完成，可以重建
    if os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0:
        print(f"🔄 检测到已存在向量库: {persist_dir}，直接加载。")
        embeddings = get_embedding(embeddings_model)
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    print("📥 首次运行，开始构建向量库...")

    # 1️⃣ 加载文件
    docs = []
    if os.path.isdir(data_path):
        print(f"📂 正在加载目录：{data_path}")
        loaders = [
            DirectoryLoader(data_path, glob="**/*.pdf",
                            loader_cls=PyPDFLoader),
            DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader),
            DirectoryLoader(data_path, glob="**/*.docx",
                            loader_cls=UnstructuredWordDocumentLoader),
            DirectoryLoader(data_path, glob="**/*.csv", loader_cls=CSVLoader),
        ]
        for loader in loaders:
            try:
                docs.extend(loader.load())
            except Exception as e:
                print(f"⚠️ 加载 {loader} 时出错: {e}")
    else:
        # 单文件
        ext = os.path.splitext(data_path)[1].lower()
        try:
            if ext == ".pdf":
                docs = PyPDFLoader(data_path).load()
            elif ext == ".txt":
                docs = TextLoader(data_path).load()
            elif ext == ".docx":
                docs = UnstructuredWordDocumentLoader(data_path).load()
            elif ext == ".csv":
                docs = CSVLoader(data_path).load()
            else:
                raise ValueError(f"❌ 不支持的文件类型: {ext}")
        except Exception as e:
            print(f"❌ 文件加载失败: {e}")
            return None

    print(f"📄 文档载入完成，共 {len(docs)} 条记录")

    # 2️⃣ 文本清洗
    for doc in docs:
        if not isinstance(doc.page_content, str):
            doc.page_content = str(
                doc.page_content) if doc.page_content else ""
        # 去除换行符、控制符
        text = re.sub(r'\s+', ' ', doc.page_content)
        # 清除过多的特殊符号
        text = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9.,!?%()（）\-–—\s]', '', text)
        doc.page_content = text.strip()
    print("✨ 文本清洗完成")

    # 3️⃣ 文本切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.1),
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    # 4️⃣ 严格过滤无效文本
    MAX_LEN = 4000  # 防止过长文本触发 tokenizer 报错
    valid_splits = []
    for d in splits:
        if not isinstance(d.page_content, str):
            continue
        content = d.page_content.strip()
        if not content:
            continue
        if len(content) > MAX_LEN:
            d.page_content = content[:MAX_LEN]  # 截断过长文本
        valid_splits.append(d)

    print(f"✅ 有效文档段落数量: {len(valid_splits)} / {len(splits)}")

    if len(valid_splits) == 0:
        raise ValueError("❌ 未检测到有效文档内容，请检查输入文件。")

    # 5️⃣ 写入向量数据库
    embeddings = get_embedding(embeddings_model)
    print(f"🚀 正在生成向量嵌入 ({embeddings_model}) ...")
    vectorstore = Chroma.from_documents(
        documents=valid_splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print("✅ 向量库构建完成并已持久化存储。")
    return vectorstore
