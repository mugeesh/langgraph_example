import os

import dotenv
from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    print("Running agent reasoning node...")
    print("State messages:", state["messages"])
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]]
    )
    print("======================================")
    print("Agent reasoning response:", response)
    print("======================================")
    return {"messages": [response]}


tool_node = ToolNode(tools)
