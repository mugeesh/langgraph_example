import dotenv
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI


load_dotenv()

deepseek_url = os.getenv("DEEPSEEK_URL")
deepseek_key = os.getenv("DEEPSEEK_API_KEY")




@tool
def triple(num: float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3

tools = [TavilySearch(max_results=1), triple]

llm = ChatOpenAI(
    model="deepseek-flash",
    temperature=0,
    api_key=deepseek_key,
    base_url=deepseek_url,
).bind_tools(tools)
