import os
import grandalf
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

from langgraph.graph.message import add_messages

from chain import generate_chain, llm, reflect_chain

load_dotenv()


class MessageGraph(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


def generate_tweet(state: MessageGraph) -> MessageGraph:
    messages = state["messages"]
    response = generate_chain.invoke({"messages": messages})
    print(">>>>>>>>>")
    print(response.content)
    print(">>>>>>>>>")
    return {"messages": [response]}


def reflect_tweet(state: MessageGraph) -> MessageGraph:
    messages = state["messages"]
    response = reflect_chain.invoke({"messages": messages})
    print(">>>>>>>>>")
    print(response.content)
    print(">>>>>>>>>")
    return {"messages": [HumanMessage(content=response.content)]}

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generate_tweet)
builder.add_node(REFLECT, reflect_tweet)
builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue, path_map={ END: END, REFLECT: REFLECT})
builder.add_edge(REFLECT, GENERATE)


graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()



if __name__ == "__main__":
    print("Hello LangGraph with ReAct Pattern")
    inputs = {
        "messages": [
            HumanMessage(
                content="""
@UIDAI
 

Enrolment No: 0014239020287520260811155538 still rejected due to missing QR on birth certificate. Child born outside India (valid Indian BC, no QR). 
Sibling’s application with same docs already approved. 

I have emailed multiple times but no reply. Please help.

                                  """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response)