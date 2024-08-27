from dotenv import load_dotenv
import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from mermeid_display import display_mermaid
from langgraph.checkpoint.sqlite import SqliteSaver

checkpoint = SqliteSaver.from_conn_string(":memory:")

load_dotenv()
# set_name in file. 
os.environ["LANGCHAIN_PROJECT"] = "LangGraph Tutorial Basic"
llm = ChatOpenAI(model="gpt-3.5-turbo")

class State(TypedDict): 
    # Messages have the type "list". The 'add message' function 
    # in the annotation defines how this state key should be updated
    # (in this case it append a message to a list, rather than overwriting them.)
    messages: Annotated[list, add_messages]


# Takes the state as an input and return update message list. 
def chatbot(state: State): 
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile(checkpointer=checkpoint)
config = {"configurable": {"thread_id": "1"}}

# call this function to display the graph. 
# display_mermaid(graph)

while True: 
    user_input = input("You : ")
    if user_input.lower() == "quit":
        print("Goodbye !") 
        break 
    for events in graph.stream({"messages": ("user", user_input)}, config): 
        for event in events.values(): 
            event["messages"][-1].pretty_print()







