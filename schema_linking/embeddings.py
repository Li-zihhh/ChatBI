"""
DashScope Embedding 封装模块。

DashScope 的 Embedding API 输入格式与 OpenAI 不同：
OpenAI:  {"input": ["text1", "text2"]}
DashScope: {"input": {"texts": ["text1", "text2"]}}

因此需要自定义封装，不能直接用 langchain_openai.OpenAIEmbeddings。
"""

import math
from typing import List

import requests
from langchain_core.embeddings import Embeddings

DASHSCOPE_EMBEDDING_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/"
    "embeddings/text-embedding/text-embedding"
)

MAX_BATCH_SIZE = 10


class DashScopeEmbeddings(Embeddings):
    """DashScope Text Embedding 封装，兼容 LangChain Embeddings 接口。

    直接调用 DashScope 原生 Embedding API，支持 text_type 参数
    （document / query）以优化 embedding 质量。
    自动分批处理，避免超出 API 单次请求文本数量限制。
    """

    def __init__(self, model: str = "text-embedding-v3", api_key: str | None = None):
        self.model = model
        self.api_key = api_key

    def _call_api(
        self, texts: List[str], text_type: str = "document"
    ) -> List[List[float]]:
        """调用 DashScope 原生 Embedding API（单次请求，不超过 MAX_BATCH_SIZE）。"""
        response = requests.post(
            DASHSCOPE_EMBEDDING_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": {"texts": texts},
                "parameters": {"text_type": text_type},
            },
            timeout=60,
        )
        if not response.ok:
            print(f"[DashScopeEmbeddings] HTTP {response.status_code}: {response.text}")
        response.raise_for_status()
        result = response.json()
        embeddings = sorted(
            result["output"]["embeddings"], key=lambda x: x["text_index"]
        )
        return [item["embedding"] for item in embeddings]

    def _call_api_batched(
        self, texts: List[str], text_type: str = "document"
    ) -> List[List[float]]:
        """分批调用 API，自动按 MAX_BATCH_SIZE 拆分。"""
        if len(texts) <= MAX_BATCH_SIZE:
            return self._call_api(texts, text_type=text_type)

        all_embeddings: List[List[float]] = []
        total_batches = math.ceil(len(texts) / MAX_BATCH_SIZE)
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i : i + MAX_BATCH_SIZE]
            batch_num = i // MAX_BATCH_SIZE + 1
            print(
                f"  [Embedding] 处理第 {batch_num}/{total_batches} 批 "
                f"（{len(batch)} 条文本）..."
            )
            all_embeddings.extend(self._call_api(batch, text_type=text_type))
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表（用于索引构建）。"""
        return self._call_api_batched(texts, text_type="document")

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本（用于检索）。"""
        return self._call_api([text], text_type="query")[0]