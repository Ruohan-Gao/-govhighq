# Load environment variables before anything else to avoid import issues
from dotenv import load_dotenv
load_dotenv()  
import os

# Set environment variables for Azure Cognitive Search field mappings --- Basically these are the fields in your Azure Search index-- this is important for the vector store to work correctly
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

# Initialize embedding model using Azure OpenAI -- this is used to create embeddings for your documents-- this model is used to convert text into vector representations
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



# Set up Azure Cognitive Search as the vector store -- this is where your documents will be stored and searched-- this is the main vector store that will be used to search for relevant documents based on user queries
vectorstore = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_API_KEY,
    index_name=AZURE_SEARCH_INDEX_NAME,
    embedding_function=embedding_model,  # Use this instead of 'embedding'
    # Specify your actual field names from the schema
    fields=["id", "content", "doc_type", "embedding"]
)









#####################  RAG SETUP USING LANGGRAPH     ##########################



from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated
import os
from langgraph.prebuilt import ToolNode

from langgraph.checkpoint.postgres import PostgresSaver
import psycopg2
from psycopg2 import pool
from langchain.tools import tool


# Define tool to be called by the LLM -- this is the function that will be called when the LLM decides to use a tool
# It searches the vector store for relevant documents based on the user's query
@tool
def search_docs(query: str) -> str:
    """Searches Azure Cognitive Search and returns relevant document chunks."""
    docs = vectorstore.similarity_search(query=query, k=3, search_type="hybrid")
    return "\n\n".join([doc.page_content for doc in docs])

tools = [search_docs]

# Define state structure for LangGraph -- this is the structure that will be used to pass messages between nodes in the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Windowing function to preserve the last N full user-assistant turns
# and ensure tool call messages are paired correctly -- this function ensures that we only keep the last N turns of conversation so that the context is not too long for the LLM

def smart_window_messages(messages, max_turns):
    if not messages:
        return []

    turns = []
    current_turn = []

    for msg in messages:
        if msg.type == "human":
            if current_turn:
                turns.append(current_turn)
            current_turn = [msg]
        else:
            current_turn.append(msg)

    if current_turn:
        turns.append(current_turn)

    # Take only the last N turns
    recent_turns = turns[-max_turns:]

    # Flatten to single list
    flat = [m for turn in recent_turns for m in turn]

    return validate_tool_pairs(flat)



# Ensure each tool message is paired with its corresponding AI message -- 

def validate_tool_pairs(messages):
    validated = []
    for i, msg in enumerate(messages):
        if msg.type == "tool":
            if validated and validated[-1].type == "ai" and hasattr(validated[-1], "tool_calls"):
                validated.append(msg)
            else:
                print("Skipping orphaned tool message.")
        else:
            validated.append(msg)
    return validated

# Main node that handles the chat logic -- this is the main node that will be called by the graph to handle the chat logic.
def chatbot_node(state: State) -> State:
    messages = state["messages"]
    max_turns = 4
    system_msgs = [msg for msg in messages if msg.type == "system"]
    user_msgs = [msg for msg in messages if msg.type != "system"]
    windowed = system_msgs + smart_window_messages(user_msgs, max_turns)
    response = llm_with_tools.invoke(windowed)
    return {"messages": messages + [response]}

# Function to route tool calls or end the conversation based on the last message -- this function checks if the last message is a tool call and routes accordingly
def route_tools(state: State):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "call_tools"
    return "end"

# Create LangGraph checkpointing tables if not already created -- this function initializes the PostgreSQL tables needed for LangGraph to work with PostgresSaver
DB_URI = os.getenv("SUPABASE_PG_CONN_STRING") 

def setup_postgres_tables():
    """Initialize PostgreSQL tables using PostgresSaver.setup()"""
    try:
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            # This is the recommended way to create tables
            checkpointer.setup()
            print("✅ PostgreSQL tables initialized successfully using .setup()")
            return True
    except Exception as e:
        print(f"❌ Failed to initialize PostgreSQL tables: {e}")
        return False

# --- CONNECTION MANAGEMENT WITH PostgresSaver ---
class ConnectionManager:
    def __init__(self):
        self._pool = None
        self._graph = None
        self._llm_with_tools = None
        self._tables_initialized = False
    
    def ensure_tables_exist(self):
        """Ensure database tables exist before using PostgresSaver"""
        if not self._tables_initialized:
            self._tables_initialized = setup_postgres_tables()
        return self._tables_initialized
    
    def get_connection_pool(self):
        if self._pool is None:
            try:
                self._pool = psycopg2.pool.SimpleConnectionPool(
                    1, 10,  # min and max connections
                    DB_URI
                )
                print("✅ Connection pool created successfully")
            except Exception as e:
                print(f"❌ Failed to create connection pool: {e}")
                raise
        return self._pool
    
    def get_llm_with_tools(self):
        if self._llm_with_tools is None:
            llm_model = AzureChatOpenAI(
                api_key=os.getenv("AZURE_LLM_API_KEY"),
                azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT"),
                deployment_name=os.getenv("AZURE_LLM_DEPLOYMENT"),
                api_version=os.getenv("AZURE_LLM_API_VERSION")
            )
            self._llm_with_tools = llm_model.bind_tools(tools)
        return self._llm_with_tools
    
    def get_graph(self):
        if self._graph is None:
            # Ensure tables exist first
            if not self.ensure_tables_exist():
                raise Exception("Failed to initialize database tables")
            
            # Create connection pool
            connection_pool = self.get_connection_pool()
            
            # Create checkpointer with pool - PostgresSaver style
            checkpointer = PostgresSaver(connection_pool)
            
            # Set up global llm_with_tools
            global llm_with_tools
            llm_with_tools = self.get_llm_with_tools()
            
            # Build graph
            tool_node = ToolNode(tools=tools)
            builder = StateGraph(State)
            
            builder.add_node("chatbot", chatbot_node)
            builder.add_node("call_tools", tool_node)
            builder.add_edge(START, "chatbot")
            builder.add_conditional_edges("chatbot", route_tools, {
                "call_tools": "call_tools",
                "end": END
            })
            builder.add_edge("call_tools", "chatbot")
            
            self._graph = builder.compile(checkpointer=checkpointer)
            print("✅ Graph compiled successfully with connection pool")
        
        return self._graph
    
    def close_pool(self):
        if self._pool:
            self._pool.closeall()
            print("✅ Connection pool closed")

def get_graph_with_context():
    """Creates graph using PostgresSaver context manager"""
    # Create LLM
    llm_model = AzureChatOpenAI(
        api_key=os.getenv("AZURE_LLM_API_KEY"),
        azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT"),
        deployment_name=os.getenv("AZURE_LLM_DEPLOYMENT"),
        api_version=os.getenv("AZURE_LLM_API_VERSION")
    )
    
    global llm_with_tools
    llm_with_tools = llm_model.bind_tools(tools)
    
    # Build graph
    tool_node = ToolNode(tools=tools)
    builder = StateGraph(State)
    
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("call_tools", tool_node)
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", route_tools, {
        "call_tools": "call_tools",
        "end": END
    })
    builder.add_edge("call_tools", "chatbot")
    
    return builder

# Global connection manager
connection_manager = ConnectionManager()


# Main function to invoke LLM and graph in production
def run_llm_with_context(user_input: str, thread_id: str):
    """Uses PostgresSaver context manager for proper connection handling"""
    builder = get_graph_with_context()
    
    try:
        # Ensure tables exist before using PostgresSaver
        if not setup_postgres_tables():
            raise Exception("Failed to initialize database tables")
        
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
            final_state = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": thread_id}}
            )
            return final_state["messages"][-1].content
    except Exception as e:
        print(f"❌ Error in run_llm_with_context: {e}")
        raise

# One-time setup function - call this once when you first deploy/setup your app
def initialize_database():
    """Call this once to set up the database tables"""
    print("🔧 Setting up database tables...")
    if setup_postgres_tables():
        print("✅ Database initialization complete")
        return True
    else:
        print("❌ Database initialization failed")
        return False

# Use context manager for proper connection handling
run_llm = run_llm_with_context

# Cleanup function to close connection pool when done
def cleanup():
    connection_manager.close_pool()