from langchain_ollama import ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def get_llm(model_name="qwen3:8b", temperature=0.1, max_tokens=1024):
    # 创建 LLM 实例
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],  # 流式输出
        max_tokens=max_tokens,
        top_p=0.7,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["<|endoftext|>"],
        device="cuda",
        port=11434  # 如果使用默认 serve 端口
    )
    return llm
