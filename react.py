import os

import dotenv
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

load_dotenv()

deepseek_url = os.getenv("DEEP_SEEK_URL")
deepseek_key = os.getenv("DEEPSEEK_API_KEY")


@tool
def triple(num: float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    print(f"Tripling the number==>>>>: {num}")
    return float(num) * 3


tools = [TavilySearch(max_results=1), triple]

llm = ChatOpenAI(
    model="deepseek-flash",
    temperature=0,
    api_key=deepseek_key,
    base_url=deepseek_url,
).bind_tools(tools)
