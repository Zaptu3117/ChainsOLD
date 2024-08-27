from typing import Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv import arxiv
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

memory = SqliteSaver.from_conn_string(":memory:")

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearchResults(max_results=2)
tools = [tool]

# We can call the LLM andcall the LLM with binded tools. 
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State): 
    return {"messages": llm_with_tools.invoke(state["messages"])}

graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile(
    checkpointer=memory, 
    # We put an interruption here. 
    interrupt_before=["tools"]
    # We can also interrupt __after__ action
    # interrupt_after = ["tools"]
)

input_user = "I'm learning LangGraph. Could you do some research on its key features and provide a summary"
config = {"configurable": {"thread_id": "1"}}

events = graph.stream({"messages": ["user", input_user]}, config, stream_mode="values")

for event in events : 
    if "messages" in event : 
        event["messages"][-1].pretty_print()


snapshot = graph.get_state(config)
print(snapshot.next)
print("foooo")
if "tools" in snapshot.next:
    # We're at the interruption point before tools
    existing_message = snapshot.values["messages"][-1]
    print(existing_message.tool_calls)

    # Decide whether to continue with tool execution
    user_decision = input("Do you want to continue with tool execution? (yes/no): ")
    
    if user_decision.lower() == "yes":
        # Continue graph execution
        events = graph.stream(None, config, stream_mode="values")
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()
    else:
        print("Tool execution skipped.")
else:
    print("Graph execution completed.")