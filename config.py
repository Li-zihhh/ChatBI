# config.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

# 直接实例化好，外部直接拿这个对象
llm = ChatOpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    model=LLM_MODEL
)
