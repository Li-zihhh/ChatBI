"""
LLM 客户端模块

负责调用大模型 API 生成 SQL。
将 LLM 调用封装为独立模块，便于后续切换不同模型或增加重试逻辑。

"""

import re
from typing import Generator
from openai import OpenAI
from config import LLM_CONFIG


class LLMClient:
    """LLM API 客户端"""

    def __init__(self, config: dict = LLM_CONFIG):
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )
        self.model = config["model"]
        self.embedding_model = config["embedding_model"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]

    def generate_sql(self, system_msg: str, prompt: str) -> str:
        """
        调用 LLM 生成 SQL（同步，一次性返回）

        Args:
            system_msg: 系统角色消息
            prompt: 完整的用户 Prompt

        Returns:
            提取后的纯 SQL 字符串
        """
        raw_output = self.generate_text(system_msg=system_msg, prompt=prompt)

        # 提取 SQL：去除 markdown 代码块标记
        sql = re.sub(r'```sql|```', '', raw_output).strip()

        return sql

    def generate_text(self, system_msg: str, prompt: str) -> str:
        """
        调用 LLM 生成普通文本（同步，一次性返回）

        Args:
            system_msg: 系统角色消息
            prompt: 完整的用户 Prompt

        Returns:
            模型返回的原始文本内容
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()

    def generate_sql_stream(self, system_msg: str, prompt: str) -> Generator[str, None, None]:
        """
        调用 LLM 流式生成 SQL（逐 chunk 产出）

        每次_yield_一段增量文本，调用方可以实时推送给前端。
        流式输出结束后，调用方需要自行拼接完整 SQL 并做 markdown 代码块清理。

        Args:
            system_msg: 系统角色消息
            prompt: 完整的用户 Prompt

        Yields:
            增量文本片段（可能包含 markdown 代码块标记，需在完整拼接后统一清理）
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            delta_content = delta.content if delta and delta.content is not None else None
            if delta_content is not None:
                yield delta_content




if __name__ == '__main__':
    llm = LLMClient()
