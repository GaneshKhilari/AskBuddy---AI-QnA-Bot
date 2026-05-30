from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import os
import shutil
import streamlit as st


## Data in st session

if"document_uploaded" not in st.session_state:
    st.session_state.document_uploaded=False
    
if "agent" not in st.session_state:
    st.session_state.agent = None
    
if"vector_store" not in st.session_state:
    st.session_state.vector_store = None
    
if"messages" not in st.session_state:
    st.session_state.messages = []
    
def process_documents(path):
    ## Load the documnets.
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    ##split the documents into chunks.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(docs)

    ## Embddings and vector db.
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
    vector_db = InMemoryVectorStore.from_documents(documents=docs, embedding=embeddings)
    st.session_state.vector_store = vector_db

    ## Create the agent- tool, llm and prompt.
    llm = ChatGroq(model="openai/gpt-oss-20b")

    @tool
    def retrieve_context(query):
        """Retrieve documents relavant to a query from the knowledge base."""
        context =""
        
        docs =vector_db.similarity_search(query =query,k=5)
        for doc in docs:
            context += doc.page_content +"\n\n"
            print("---- Retrieved Context ----")
            print(context)
        return context

    
    system_prompt = """
        You are a strict document-based assistant.

        Rules:
        1. ALWAYS call the 'retrieve_context' tool.
        2. Answer ONLY from retrieved context.
        3. If context is "NO_RELEVANT_CONTEXT", say:
            "I don't know based on the provided document."
        4. Do NOT use outside knowledge.
        5. Do NOT guess.

        Keep answers precise and factual.
"""


    memory = InMemorySaver()


    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer = memory
    )
    st.session_state.agent = agent
    st.session_state.document_uploaded = True
    

if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(
        label="Select PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:
        with st.spinner("Processing..."):

            path = "C:\\GenAI\\apps\\doc_files\\"

            # STEP 1: Delete old folder (and all PDFs inside it)
            if os.path.exists(path):
                shutil.rmtree(path)

            # STEP 2: Create fresh empty folder
            os.makedirs(path)

            # STEP 3: Save only newly uploaded PDFs
            for file in uploaded:
                file_path = os.path.join(path, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())

            # STEP 4: Reset chat + vector DB (IMPORTANT)
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.session_state.agent = None

            # STEP 5: Process documents
            process_documents(path)

            st.rerun()
if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)
    query = st.chat_input("Ask a question about the documents:")
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages":[{"role":"user","content":query}]},{"configurable": {"thread_id": 1}}
        
        )
        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai","content":answer})