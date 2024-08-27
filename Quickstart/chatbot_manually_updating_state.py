from typing import Annotated
from dotenv import load_dotenv
from mermeid_display import display_mermaid

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, ToolMessage

load_dotenv()

class State(TypedDict): 
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatOpenAI(model="gpt-3.5-turbo")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State): 
    message = state["messages"]
    response = llm_with_tools.invoke(message)
    return {"messages": response}


graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot", 
    tools_condition
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
memory = SqliteSaver.from_conn_string(":memory:")
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["tools"])

user_input = "I'm learning langgraph. Could you make some internet research and summarize it for me ?"
config = {"configurable": {"thread_id": "1"}}

events = graph.stream({"messages": user_input}, config, stream_mode="values")

for event in events: 
    if "messages" in event: 
        event["messages"][-1].pretty_print()

snapshot = graph.get_state(config)
existing_message = snapshot.values["messages"][-1]
existing_message.pretty_print()


# This will inject a human prompt as a LLM response. 
answer = (
    "LangGraph is a library for building statefull, multi-actor applications with LLM."
)
new_messages = [
    ToolMessage(       
        content=answer,        
        tool_call_id=existing_message.tool_calls[0]["id"], 
    ), 
    AIMessage(content=answer) 
]
new_messages[-1].pretty_print()
graph.update_state(config, {"messages": [AIMessage(content="I'm an AI expert, bitch !")]}, as_node="chatbot")
print("\n\n Last 2 messages :")
print(graph.get_state(config).values["messages"][-2:])
display_mermaid(graph)


# How to replace a message. 
config = {"configurable": {"thread_id": "2"}}
events = graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values")
print("Events :", events)
for event in events: 
    if "messages" in events: 
        event["messages"][-1].pretty_print()
snapshot = graph.get_state(config)
existing_message = snapshot.values["messages"][-1]
print("Original")
print("Message ID", existing_message.id)
print(existing_message.tool_calls[0])
new_tool_call = existing_message.tool_calls[0].copy()
new_tool_call["args"]["query"] = "LangGraph human-in-the-loop workflow"
new_message = AIMessage(
    content=existing_message.content,
    tool_calls=[new_tool_call],
    # Important! The ID is how LangGraph knows to REPLACE the message in the state rather than APPEND this messages
    id=existing_message.id,
)

print("Updated")
print(new_message.tool_calls[0])
print("Message ID", new_message.id)
graph.update_state(config, {"messages": [new_message]})

print("\n\nTool calls")
graph.get_state(config).values["messages"][-1].tool_calls
events = graph.stream(None, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

events = graph.stream(
    {
        "messages": (
            "user",
            "Remember what I'm learning about?",
        )
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()