"""
LLM 客户端模块

封装 LangChain ChatOpenAI,提供 SQL 生成能力。
支持兼容 OpenAI API 的各种服务(阿里云 DashScope / 智谱 / 本地 Ollama 等)。
"""

import re

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_models.openai import ChatOpenAI

from config import LLM_CONFIG


class LLMError(Exception):
    """LLM 调用失败的统一异常"""


class LLMClient:
    """LLM 客户端"""

    def __init__(self, config: dict = None):
        self.config = config if config is not None else LLM_CONFIG

        api_key = self.config.get("api_key")
        base_url = self.config.get("base_url")
        model = self.config.get("model")

        if not api_key:
            raise LLMError(
                "未配置 API Key,请在 .env 或环境变量中设置 OPENAI_API_KEY"
            )

        if not base_url:
            raise LLMError(
                "未配置 API Base URL,请在 .env 或环境变量中设置 OPENAI_BASE_URL"
            )

        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=self.config.get("max_tokens", 4000),
            temperature=self.config.get("temperature", 0.2),
        )

    def generate_sql(self, system_prompt: str, prompt: str) -> str:
        """
        调用 LLM 生成 SQL

        Args:
            system_prompt: 系统提示词(角色设定)
            prompt: 用户提示词(具体任务)

        Returns:
            生成的 SQL 字符串
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
        except Exception as e:
            error_text = str(e)

            if "401" in error_text or "Unauthorized" in error_text:
                raise LLMError("API Key 无效或已过期,请检查 OPENAI_API_KEY")
            if "404" in error_text or "Not Found" in error_text:
                raise LLMError(
                    f"API 端点不存在,请检查 OPENAI_BASE_URL 是否正确,当前: {self.config.get('base_url')}"
                )
            if "Connection" in error_text or "timeout" in error_text.lower():
                raise LLMError(
                    f"无法连接到 API 服务,请检查网络和 base_url: {self.config.get('base_url')}"
                )

            raise LLMError(f"LLM 调用失败: {error_text}")

        raw_output = response.content.strip()

        if raw_output.startswith("<!DOCTYPE html") or raw_output.startswith("<html"):
            raise LLMError(
                "API 返回了 HTML 页面而不是 JSON 响应,"
                f"base_url 可能配置错误: {self.config.get('base_url')}"
            )

        sql = re.sub(r"```sql|```", "", raw_output).strip()
        return sql