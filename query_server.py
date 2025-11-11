from flask import Flask, request, jsonify, Response
from Chat_QA_chain import Chat_QA_chain
import json
from flask_cors import CORS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from flask import Response, stream_with_context
from StreamRunnable import StreamableRunnable
from model_to_llm import get_llm
import re

app = Flask(__name__)
CORS(app)  # 默认允许所有域名访问

streamable_chain = StreamableRunnable(get_llm())

def query(question: str, qa_chain, show_sources: bool = False):
    """执行问答，增加异常捕获"""
    try:
        # 调用 QA 链
        print("执行问答")
        # 在调用时传入回调
        result = qa_chain.invoke({"query": question})
        answer = result.get('result', "文档中未找到答案")
        print("回答完成")
        sources = []
        # 收集参考来源
        if show_sources and 'source_documents' in result:
            for doc in result['source_documents']:
                sources.append({
                    "source": doc.metadata.get('source', 'N/A'),
                    "content": doc.page_content
                })

    except (IndexError, KeyError, TypeError) as e:
        # 捕获 refine_chain 内部 docs 为空等异常
        answer = "文档中未找到答案"
        sources = []

    return answer, sources

def generate_stream(question: str, show_sources: bool):
    """生成流式响应"""
    try:
        # 使用 stream 方法直接获取生成器       
        for chunk in qa_chain.stream({"query": question}):
            # 处理每个 chunk
            if hasattr(chunk, 'get'):
                token = chunk.get('answer', '') or chunk.get('text', '') or chunk.get('result', '')
            else:
                token = str(chunk)
            
            if token:
                yield f"data: {json.dumps({'token': token})}\n\n"
        
        # 获取最终结果用于来源显示
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.route("/query", methods=["POST"])
def handle_query():
    """接收 POST 请求，返回问答结果"""
    print("接收到post请求")

    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        show_sources = data.get("show_sources", False)
        stream = data.get("stream", False)

        if not question:
            return jsonify({"error": "问题不能为空"}), 400
        if not stream:
            answer, sources = query(question, qa_chain, show_sources)
            return jsonify({
                "question": question,
                "answer": answer,
                "sources": sources
            })
        else:
            # 流式输出 NDJSON，每行一个 JSON token
            def gen():
                for token in streamable_chain.stream(question):
                    # 去掉多余空格和换行
                    clean_token = re.sub(r'\s+', ' ', token).strip()
                    if clean_token:
                        yield json.dumps({"token": clean_token}, ensure_ascii=False) + "\n"
                yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"

            return Response(stream_with_context(gen()),
                            content_type="application/x-ndjson; charset=utf-8")
        
    except Exception as e:
        return jsonify({"error": f"处理请求时发生错误: {str(e)}"}), 500


@app.route("/", methods=['GET'])
def index():
    return jsonify({"message": "Hello World"})
    

if __name__ == "__main__":
    file_path = "./data"  # Replace with your actual file path
    chat_qa_chain = Chat_QA_chain(file_path)
    qa_chain = chat_qa_chain.build_qa_chain()
    app.run(host="0.0.0.0", port=8081)
