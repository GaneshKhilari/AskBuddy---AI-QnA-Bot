from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "agent" not in st.session_state:
    st.session_state.agent = None

PDF_FOLDER = r"C:\GenAI\apps\doc_files"
CHROMA_PATH = "./chroma_db"

os.makedirs(PDF_FOLDER, exist_ok=True)

# EMBEDDINGS + VECTOR DB
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = get_embeddings()

@st.cache_resource
def get_vector_db(_embeddings):
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=_embeddings
    )

vector_db = get_vector_db(embeddings)   


st.session_state.vector_store = vector_db

# PDF PROCESSING
def process_pdf(file_path):

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    st.session_state.vector_store.add_documents(chunks)
    
# TOOL
@tool
def retrieve_context(query: str):
    """Retrieve relevant information from uploaded PDF documents."""
    print("Tool Called")
    print(vector_db)

    docs = vector_db.similarity_search(
        query=query,
        k=5
    )

    if not docs:
        return "NO_RELEVANT_CONTEXT"

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return context

# AGENT

@st.cache_resource
def get_agent(_vector_db):
    

    llm = ChatGroq(model="llama-3.1-8b-instant")
    system_prompt = """
You are a strict document-based assistant.

Rules:
1. ALWAYS use the retrieve_context tool.
2. Answer ONLY from retrieved context.
3. If context is NO_RELEVANT_CONTEXT, say:
   "I don't know based on the provided documents."
4. Never use outside knowledge.
5. Never guess.
"""

    memory = InMemorySaver()

    return create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

st.session_state.agent = get_agent(vector_db)
# FILE UPLOADER


uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    with st.spinner("Processing PDFs..."):

        for file in uploaded_files:

            file_path = os.path.join(
                PDF_FOLDER,
                file.name
            )

            # Skip already processed files
            if os.path.exists(file_path):
                continue

            with open(file_path, "wb") as f:
                f.write(file.getvalue())

            process_pdf(file_path)

        st.success("PDFs processed successfully.")

# CHAT HISTORY
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# CHAT INPUT

query = st.chat_input(
    "Ask a question about your PDFs..."
)

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    response = st.session_state.agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        },
        {
            "configurable": {
                "thread_id": "pdf_chat"
            }
        }
    )

    answer = response["messages"][-1].content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )