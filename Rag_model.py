from dotenv import load_dotenv
load_dotenv()  # ✅ MUST come before importing langchain

import os
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "embedding"
os.environ["AZURESEARCH_FIELDS_CONTENT"] = "content"
os.environ["AZURESEARCH_FIELDS_ID"] = "id"
os.environ["AZURESEARCH_FIELDS_METADATA"] = "doc_type"

from langchain_openai import AzureOpenAIEmbeddings
from config import (
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_INDEX_NAME
)

# Initialize the embedding model
embedding_model = AzureOpenAIEmbeddings(
    deployment=AZURE_OPENAI_DEPLOYMENT,
    model=AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    chunk_size=1
)

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings



# Correct setup for Azure Cognitive Search
vectorstore = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_API_KEY,
    index_name=AZURE_SEARCH_INDEX_NAME,
    embedding_function=embedding_model,  # Use this instead of 'embedding'
    # Specify your actual field names from the schema
    fields=["id", "content", "doc_type", "embedding"]
)




# # Get retriever
# retriever = vectorstore.as_retriever( search_type="similarity",search_kwargs={"k": 3})

# Run query
# query = "What contracts are expiring this month?"
# docs = vectorstore.similarity_search(
#     query="What contracts are expiring this month?",
#     k=3,
#     search_type="similarity"
# )

# print(docs[0].page_content)




#####################  RRRAAAGGGGGG    ##########################



from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated
import os

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver 




# Create the connection manually
conn = sqlite3.connect("chat_memory.db",check_same_thread=False)
checkpointer = SqliteSaver(conn)

# 2. Set up LLM (chat model)
llm_model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_LLM_API_KEY"),              
    azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT"),      
    deployment_name=os.getenv("AZURE_LLM_DEPLOYMENT"),   
    api_version=os.getenv("AZURE_LLM_API_VERSION")
)

# 3. Define LangGraph-compatible state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 4. Define retrieve node
from langchain.tools import tool

@tool
def search_docs(query: str) -> str:
    """Searches Azure Cognitive Search and returns relevant document chunks."""
    docs = vectorstore.similarity_search(query=query, k=3, search_type="hybrid")
    return "\n\n".join([doc.page_content for doc in docs])

tools = [search_docs]
llm_with_tools = llm_model.bind_tools(tools)



# def chatbot_node(state: State) -> State:#
    
    # response = llm_with_tools.invoke(state["messages"])#
#     return {"messages": [response]}#


def chatbot_node(state: State) -> State:
    # Implement windowed message history here
    messages = state["messages"]
    
    # Keep only the last N conversation turns (adjust as needed)
    max_turns = 4  # Keep last 4 conversation turns
    
    # Always keep the system message if it exists
    windowed_messages = []
    system_messages = [msg for msg in messages if msg.type == "system"]
    other_messages = [msg for msg in messages if msg.type != "system"]
    
    # Add system messages first
    windowed_messages.extend(system_messages)
    
    # Smart windowing that preserves tool call/tool response pairs
    windowed_messages.extend(smart_window_messages(other_messages, max_turns))
    
    response = llm_with_tools.invoke(windowed_messages)
    return {"messages": [response]}

def smart_window_messages(messages, max_turns):
    """
    Intelligently window messages while preserving tool call/tool response pairs.
    A 'turn' is defined as: HumanMessage -> AssistantMessage (+ optional ToolMessages)
    """
    if not messages:
        return []
    
    # Group messages into conversation turns
    turns = []
    current_turn = []
    
    for msg in messages:
        if msg.type == "human" and current_turn:
            # Start of new turn, save the previous one
            turns.append(current_turn)
            current_turn = [msg]
        else:
            current_turn.append(msg)
    
    # Add the last turn
    if current_turn:
        turns.append(current_turn)
    
    # Keep only the last N turns
    if len(turns) > max_turns:
        windowed_turns = turns[-max_turns:]
    else:
        windowed_turns = turns
    
    # Flatten back to a list of messages
    windowed_messages = []
    for turn in windowed_turns:
        windowed_messages.extend(turn)
    
    # Final validation: ensure tool messages have their corresponding tool calls
    validated_messages = validate_tool_pairs(windowed_messages)
    
    return validated_messages

def validate_tool_pairs(messages):
    """
    Ensure that every ToolMessage has a corresponding AssistantMessage with tool_calls.
    Remove orphaned tool messages.
    """
    validated = []
    i = 0
    
    while i < len(messages):
        msg = messages[i]
        
        if msg.type == "tool":
            # Check if the previous message is an assistant message with tool_calls
            if (validated and 
                validated[-1].type == "ai" and 
                hasattr(validated[-1], 'tool_calls') and 
                validated[-1].tool_calls):
                # Valid tool response, keep it
                validated.append(msg)
            else:
                # Orphaned tool message, skip it
                print(f"Skipping orphaned tool message: {msg}")
                pass
        else:
            # Non-tool message, always keep
            validated.append(msg)
        
        i += 1
    
    return validated

from langgraph.prebuilt import ToolNode

# FIXED: Custom tools condition function#
def route_tools(state: State):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    last_message = state["messages"][-1]
    
    # Check if the last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "call_tools"
    else:
        return "end"  # Return string "end", not END constant

# 1. Tool Node
tool_node = ToolNode(tools=tools)

# 2. Build Graph
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("call_tools", tool_node)

# 3. Add entry point
graph_builder.add_edge(START, "chatbot")

# 4. FIXED: Conditional branching with custom route_tools function
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {
        "call_tools": "call_tools",  # When route_tools returns "call_tools"
        "end": END                   # When route_tools returns "end"
    }
)

# 5. After using tool, return to chatbot for final answer
graph_builder.add_edge("call_tools", "chatbot")

# 6. Compile with checkpointer
graph = graph_builder.compile(checkpointer=checkpointer)



      

def run_llm(user_input: str, thread_id: str):
    final_state = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    return final_state["messages"][-1].content


