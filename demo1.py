from langchain_ollama import ChatOllama

chat_model = ChatOllama(model="qwen3:8b")

response = chat_model.invoke("你好我是张三?")

print(response)  # 输出模型回复

response = chat_model.invoke("我叫什么名字呢?")

print(response)  # 输出模型回复