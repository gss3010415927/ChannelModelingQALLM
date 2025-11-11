from queue import Queue
from threading import Thread
from typing import Iterator, Optional, Any
from langchain.schema import BaseOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import Runnable

class StreamableRunnable(Runnable):
    """支持逐 token 流式输出的 Runnable"""

    def __init__(self, llm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm  # 底层 LLM 对象，必须支持 streaming=True

    def invoke(self, input: Any, config: Optional[Any] = None, **kwargs):
        # 默认同步调用
        return self.llm.predict(input, **kwargs)

    def stream(self, input: Any, config: Optional[Any] = None, **kwargs) -> Iterator[str]:
        """重写 stream 方法，实现逐 token 输出"""
        q = Queue()

        class StreamHandler(BaseCallbackHandler):
            def on_llm_new_token(self, token: str, **kwargs):
                q.put(token)

            def on_llm_end(self, response, **kwargs):
                q.put(None)  # 标记结束

        def run_llm():
            try:
                self.llm.predict(input, callbacks=[StreamHandler()], **kwargs)
            except Exception as e:
                q.put(f"__ERROR__:{str(e)}")
                q.put(None)

        Thread(target=run_llm).start()

        while True:
            token = q.get()
            if token is None:
                break
            elif isinstance(token, str) and token.startswith("__ERROR__"):
                raise Exception(token.replace("__ERROR__:", ""))
            else:
                yield token  # 每次 yield 一个 token
