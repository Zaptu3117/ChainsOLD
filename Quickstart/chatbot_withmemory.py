from dotenv import load_dotenv
import os
from typing import Annotated

# my own functions. 
from mermeid_display import display_mermaid

# langchain library importation.
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

# langgraph library importation
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "LangGraph Chat with memory"

# This is not linked to a file, so there is no persistence between script. 
# Need to link to file to store. 
# memory = SqliteSaver.from_conn_string("sqlite:///chat_memory.db")
memory = SqliteSaver.from_conn_string(":memory:")

class State(TypedDict): 
    messages:  Annotated[list, add_messages]

def chatbot(state: State): 
    return {"messages": llm_with_tools.invoke(state["messages"])}

if __name__ == "__main__": 
    graph_builder = StateGraph(State)
    tool = TavilySearchResults(max_results=2)
    tools = [tool]

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    llm_with_tools = llm.bind_tools(tools)

    #Building the graph. 
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    graph = graph_builder.compile(checkpointer=memory)
    display_mermaid(graph)

    config = {"configurable": {"thread_id": "1"}}

    # stream is equivalent to invoke. 
    user_input = "Hello my name is Will !"

    # important to understand 
    # Can call the graph multiple time and the thread keep constant.
    #  
    events = graph.stream(
        {"messages": ["user", user_input]}, config, stream_mode="values")

    for event in events: 
        event["messages"][-1].pretty_print()

    user_input = "What's my name ?"

    events = graph.stream(
        {"messages": ["user", user_input]}, config, stream_mode="values")

    for event in events: 
        event["messages"][-1].pretty_print()

    #print complet graph execution.
    snapshot = graph.get_state(config)
    print(snapshot)


