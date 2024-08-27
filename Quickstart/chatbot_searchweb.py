from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from typing import Annotated, Any, Literal
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
import json
from langchain_core.messages import ToolMessage, BaseMessage, AIMessage
from mermeid_display import display_mermaid
import os

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "LangGraph Chat with search engine"

class State(TypedDict):
    messages: Annotated[list, add_messages]

class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
    
    def __call__(self, state: dict) -> Any:
        if messages := state.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        
        outputs = []
        print("This is the OpenAI message", message)
        
        for tool_call in message.additional_kwargs.get('tool_calls', []):
            tool_result = self.tools_by_name[tool_call["function"]["name"]].invoke(
                json.loads(tool_call["function"]["arguments"])
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["function"]["name"],
                    tool_call_id=tool_call["id"]
                )
            )
        
        return {"messages": outputs}

def route_tools(state: State) -> Literal["tools", "__end__"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No message found in input state to tool_edge: {state}")
    
    if isinstance(ai_message, AIMessage) and ai_message.additional_kwargs.get('tool_calls'):
        return "tools"
    return "__end__"

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

if __name__ == "__main__":
    tool = TavilySearchResults(max_results=2)
    tools = [tool]
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    graph_builder = StateGraph(State)
    llm_with_tools = llm.bind_tools(tools)
    
    graph_builder.add_node("chatbot", chatbot)
    tool_nodes = BasicToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_nodes)
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", "__end__": "__end__"}
    )
    
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    graph = graph_builder.compile()
    display_mermaid(graph)
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        for event in graph.stream({"messages": [("human", user_input)]}):
            for value in event.values():
                if isinstance(value["messages"][-1], BaseMessage):
                    print("Assistant:", value["messages"][-1].content)
                else:
                    print("Tool Result:", value["messages"][-1])