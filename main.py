import dotenv
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


deepseek_url=os.getenv("DEEP_SEEK_URL")
deepseek_key=os.getenv("DEEPSEEK_API_KEY")


llm = ChatOpenAI(
    model="deepseek-flash",
    temperature=0,
    api_key=deepseek_key,
    base_url=deepseek_url,
)



if __name__ == "__main__":
    print("Hello ReAct langgraph with function calling!")

