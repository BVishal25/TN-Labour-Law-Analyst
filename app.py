import streamlit as st
import os
from pathlib import Path

# --- LlamaIndex Imports and Setup ---
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.objects import ObjectIndex
from llama_index.core.agent import AgentRunner
from typing import List, Optional, Tuple

# --- Configuration ---
# Directory for your PDF documents
DATA_DIR = Path("data/legal_docs")
# Directory to store persisted LlamaIndex components (one for each document's index)
PERSIST_BASE_DIR = "./document_indexes"

# Access the API key using st.secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Error: Google AI API Key not found in .streamlit/secrets.toml or Streamlit Cloud secrets.")
    st.stop()

# Configure LlamaIndex settings
Settings.llm = Gemini(model="gemini-1.5-flash-latest", api_key=GOOGLE_API_KEY)
Settings.embed_model = GeminiEmbedding(model_name="text-embedding-004", api_key=GOOGLE_API_KEY)
Settings.chunk_size = 1024
Settings.chunk_overlap = 200

# List of Department Labour and Employment Acts (25) - NOW IN GLOBAL SCOPE
document_info = [
    {"path": DATA_DIR / "The Employees Compensation Act,1923(CENTRAL ACT).pdf", "name": "Employees Compensation Act 1923"},
    {"path": DATA_DIR / "The Trade Unions Act (CENTRAL ACT).pdf", "name": "Trade Unions Act 1926"},
    {"path": DATA_DIR / "The Payment Of Wages Act (CENTRAL ACT).pdf", "name": "Payment of Wages Act 1936"},
    {"path": DATA_DIR / "The Industrial Employment Standing Orders Act (CENTRAL ACT).pdf", "name": "Industrial Employment Standing Orders Act 1946"},
    {"path": DATA_DIR / "The Industrial Disputes Act,1947 (CENTRAL ACT).pdf", "name": "Industrial Disputes Act 1947"},
    {"path": DATA_DIR / "The Tamil Nadu Shops and Establishments Act.pdf", "name": "Shops and Establishments Act 1947"},
    {"path": DATA_DIR / "The Minimum Wages Act 1948 (CENTRAL ACT).pdf", "name": "Minimum Wages Act 1948"},
    {"path": DATA_DIR / "The Factories Act(CENTRAL ACT).pdf", "name": "Factories Act 1948"},
    {"path": DATA_DIR / "The Plantations Labour Act 1951(CENTRAL ACT).pdf", "name": "Plantations Labour Act 1951"},
    {"path": DATA_DIR / "The Tamil Nadu Catering Establishments Act.pdf", "name": "Catering Establishments Act 1958"},
    {"path": DATA_DIR / "The Tamil Nadu Industrial Establishments (National,Festivel and Special Holidays) Act.pdf", "name": "Industrial Establishments (National,Festivel and Special Holidays) Act 1958"},
    {"path": DATA_DIR / "The Motors Transport Workers Act (CENTRAL ACT).pdf", "name": "Transport Workers Act 1961"},
    {"path": DATA_DIR / "The Maternity Benefit Act (CENTRAL ACT).pdf", "name": "Maternity Benefit Act 1961"},
    {"path": DATA_DIR / "The Beedi and Cigar Workers (Conditions of Employment) Act (CENTRAL ACT).pdf", "name": "Beedi and Cigar Workers Act 1966"},
    {"path": DATA_DIR / "The Contract Labour (Regulation And Abolition) Act,1970 (CENTRAL ACT).pdf", "name": "Contract Labour Act 1970"},
    {"path": DATA_DIR / "The Payment of Gratuity Act 1972 (CENTRAL ACT).pdf", "name": "Payment of Gratuity Act 1972"},
    {"path": DATA_DIR / "The Tamil Nadu Labour Welfare Fund Act.pdf", "name": "Labour Welfare Fund Act 1972"},
    {"path": DATA_DIR / "The BONDED LABOUR SYSTEM (ABOLITION) ACT, 1976.pdf", "name": "Bonded Labour System Act 1976"},
    {"path": DATA_DIR / "The Inter-State Migrant Workmen (Regulation of Employment and Conditions of Service) Act,1979 (CENTRAL ACT).pdf", "name": "Inter-State Migrant Workmen Act 1979"},
    {"path": DATA_DIR / "The Tamil Nadu Payment of Subsistence Allowance Act.pdf", "name": "Payment of Subsistence Allowance Act 1981"},
    {"path": DATA_DIR / "The Tamil Nadu Industrial Establishments (Conferment Of Permanent Status To Workmen) Act.pdf", "name": "Industrial Establishments (Conferment Of Permanent Status To Workmen) Act 1981"},
    {"path": DATA_DIR / "The Tamil Nadu Manual Workers (Regulation Of Employment And Conditions Of Work) Act,1982.pdf", "name": "Manual Workers Act 1982"},
    {"path": DATA_DIR / "The Building And Other Construction Workers (Regulation Of Employment And Conditions Of Service) Act (CENTRAL ACT).pdf", "name": "Building And Other Construction Workers Act 1996"},
    {"path": DATA_DIR / "The CHILD AND ADOLESCENT LABOUR (PROHIBITION AND REGULATION) ACT, 1986.pdf", "name": "Child and Adolescent (Prohibition and Regulation) Act 1986"},
    {"path": DATA_DIR / "The Tamil Nadu Building and Construction Workers (Conditions of Employment and Miscellaneous Provisions) Repeal Act, 2023.pdf", "name": "Building and Construction Workers Repeal Act 2023"},
]


# Prepare the documents and Tools
# Modified to include persistence for each document's vector and summary index
def get_doc_tools(
    file_path: str,
    name: str,
    persist_dir: Path, # Added persist_dir argument
) -> Tuple[FunctionTool, QueryEngineTool]:
    """Get vector query and summary query tools from a document, with persistence."""

    doc_persist_path = persist_dir / name.replace(" ", "_").lower() # Unique folder for each doc's index

    # Check if this specific document's index is already persisted
    if not doc_persist_path.exists():
        # Load documents for this specific PDF
        st.info(f"Indexing '{name}' for the first time...")
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        
        # Add metadata to documents (optional but good for tracking)
        for doc in documents:
            doc.metadata["document_name"] = name
            doc.metadata["file_path"] = str(file_path) # Convert to string for metadata

        # Split documents into nodes (chunks)
        splitter = SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)
        nodes = splitter.get_nodes_from_documents(documents)

        # Create VectorStoreIndex and its query tool
        vector_index = VectorStoreIndex(nodes)
        vector_index.storage_context.persist(persist_dir=doc_persist_path / "vector_store")

        # Create SummaryIndex and its query tool
        summary_index = SummaryIndex(nodes)
        summary_index.storage_context.persist(persist_dir=doc_persist_path / "summary_store")
        st.success(f"'{name}' indexed and saved!")
    else:
        st.info(f"Loading index for '{name}' from disk...")
        # Load existing indexes
        vector_storage_context = StorageContext.from_defaults(persist_dir=doc_persist_path / "vector_store")
        vector_index = load_index_from_storage(vector_storage_context)
        
        summary_storage_context = StorageContext.from_defaults(persist_dir=doc_persist_path / "summary_store")
        summary_index = load_index_from_storage(summary_storage_context)
        st.success(f"'{name}' loaded from disk.")
    
    # Define vector_query function (remains the same)
    def vector_query(
        query: str, 
        page_numbers: Optional[List[str]] = None
    ) -> str:
        """Use to answer questions over the specific legal document related to {name}.
        
        Useful if you have specific questions over the document's content.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.
        
        Args:
            query (str): The string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages (e.g., ['1', '5']). 
                Leave as NONE if we want to perform a vector search over all pages.
        """
        page_numbers = page_numbers or []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2, # Retrieve top 2 relevant chunks
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            ) if page_numbers else None # Apply filters only if page_numbers are provided
        )
        response = query_engine.query(query)
        return str(response) # Ensure response is string

    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name.replace(' ', '_').lower()}", # Name for the agent
        fn=vector_query,
        description=(
            f"Use to answer specific, detailed questions over the legal document titled '{name}'. "
            f"Good for queries about sections, clauses, definitions, penalties, or very specific information. "
            f"Do NOT use this for summarization. Always specify the 'query' argument."
        )
    )
    
    # Define summary_query_engine and summary_tool (remains the same)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize", # Good for summarizing across nodes
        use_async=True, # For faster processing
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name.replace(' ', '_').lower()}", # Name for the agent
        query_engine=summary_query_engine,
        description=(
            f"Useful for high-level summarization and overview questions related to the legal document titled '{name}'. "
            f"Use this when a user asks for a general summary, introduction, or key takeaways of the document. "
            f"Do NOT use this for specific factual lookup."
        ),
    )

    return vector_query_tool, summary_tool


# --- Load all documents and create tools (using Streamlit's cache) ---
@st.cache_resource # Cache the result of this function to avoid re-indexing on every run
def load_and_prepare_tools(doc_info_list): # Now accepts document_info as an argument
    all_tools = []
    document_status = {}
    
    # Ensure data and persistence directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    Path(PERSIST_BASE_DIR).mkdir(parents=True, exist_ok=True)

    # Display initial loading message only once at the beginning of the cached function
    st.markdown("---") # Separator for clarity
    st.info(f"Starting document processing... This will create/load indexes for {len(doc_info_list)} documents.")
    st.markdown("---")

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, doc_info in enumerate(doc_info_list): # Use doc_info_list here
        file_path = doc_info["path"]
        doc_name = doc_info["name"]
        
        progress = (i + 1) / len(doc_info_list)
        progress_bar.progress(progress)
        status_text.text(f"Processing document {i+1}/{len(doc_info_list)}: {doc_name}...")

        if file_path.exists():
            try:
                vector_tool, summary_tool = get_doc_tools(str(file_path), doc_name, Path(PERSIST_BASE_DIR)) # Pass persist_dir
                all_tools.extend([vector_tool, summary_tool])
                document_status[doc_name] = "Loaded Successfully"
            except Exception as e:
                document_status[doc_name] = f"Error loading: {e}"
                st.error(f"Error processing {doc_name}: {e}")
                # Print to console for detailed debugging
                print(f"Error processing {doc_name}: {e}")
        else:
            document_status[doc_name] = "File Not Found"
            st.warning(f"Document not found: {file_path}. Please ensure it's in the '{DATA_DIR}' directory.")
            print(f"WARNING: Document not found: {file_path}") # Also print to console

    progress_bar.empty() # Clear the progress bar when done
    status_text.empty() # Clear the status text

    if not all_tools:
        st.error("No documents were successfully loaded. The AI agent cannot be initialized.")
        return [], {} # Return empty lists if no tools
    else:
        st.success(f"Successfully processed {len(all_tools) // 2} legal documents!")
        return all_tools, document_status

# Call the cached function to get tools and status, passing document_info
all_tools, document_loading_status = load_and_prepare_tools(document_info)

# Initialize the agent after tools are loaded
agent = None
if all_tools:
    obj_index = ObjectIndex.from_objects(all_tools)
    obj_retriever = obj_index.as_retriever(similarity_top_k=5)
    agent = AgentRunner.from_llm(
        llm=Settings.llm,
        tools=all_tools,
        verbose=True, # Keep this True for debugging on Hugging Face logs
        system_prompt=(
            "You are an expert legal document analyst specializing in Tamil Nadu Labour and Employment laws and regulations. "
            "Your task is to accurately and comprehensively answer user questions by leveraging the provided legal documents. "
            "Prioritize using the 'vector_tool' for specific, factual questions and the 'summary_tool' for general overviews or introductions to a document. "
            "Always clearly identify the relevant legal document(s) used in your answer. "
            "If a question requires information from multiple documents, use the relevant tools sequentially, comparing and synthesizing information as needed. "
            "If you cannot find a definitive answer in the provided documents, state that clearly and politely mention that the information might not be within the scope of the loaded documents. "
            "Provide direct answers where possible, and summarize concisely when asked."
        )
    )
else:
    st.error("The AI agent cannot be initialized because no tools were loaded successfully.")


# --- Streamlit UI ---
st.set_page_config(
    page_title="Tamil Nadu Labour and Employment Law Analyst 👷‍♀️⚖️",
    page_icon="📚",
    layout="wide",
)

st.title("Tamil Nadu Labour and Employment Law Analyst 👷‍♀️⚖️")
st.markdown(
    "Ask questions about the **Tamil Nadu Labour and Employment laws and regulations**. The system can answer specific questions or provide summaries from the loaded documents."
)

st.sidebar.header("Loaded Documents Status")
st.sidebar.markdown(f"Total documents configured: **{len(document_info)}**") # NOW document_info is defined globally
for doc_name, status in document_loading_status.items():
    st.sidebar.markdown(f"- **{doc_name}**: `{status}`")

# Chat interface
if agent:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a question about Tamil Nadu Labour and Employment Laws:",
                                placeholder="e.g., What are the legally mandated provisions for paid leave in Tamil Nadu?",
                                key="chat_input")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = agent.chat(user_query)
                    st.markdown(response.response)
                    # You can also display source nodes for transparency if desired
                    if response.source_nodes:
                        st.subheader("Sources:")
                        for node in response.source_nodes:
                            st.write(f"- Document: {node.metadata.get('document_name', 'N/A')}, Page: {node.metadata.get('page_label', 'N/A')}")
                            st.code(node.get_content()[:200] + "...", language="text") # Show a snippet
                except Exception as e:
                    st.error(f"An error occurred: {e}. Please try again or rephrase your question.")
                    response = None
                    import traceback
                    traceback.print_exc() 

            if response:
                st.session_state.messages.append({"role": "assistant", "content": response.response})