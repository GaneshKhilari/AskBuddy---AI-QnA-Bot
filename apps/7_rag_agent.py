from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
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
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = InMemoryVectorStore.from_documents(documents=docs, embedding=embeddings)
    st.session_state.vector_store = vector_db

    ## Create the agent- tool, llm and prompt.
    llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
    @tool
    def retrieve_context(query):
        """Retrieve documents relavant to a query from the knowledge base."""
        results = vector_db.similarity_search_with_score(query, k=4)
        context =""
        
        for doc, score in results:
            if score < 0.7:   # 🔥 filter irrelevant chunks
                source = doc.metadata.get("source", "unknown")
                context += f"{doc.page_content[:400]}\n(Source: {source})\n\n"  # 🔥 limit size
        return context if context.strip() else "NO_RELEVANT_CONTEXT"

    
    system_prompt = """
        You are a strict document-based assistant.

        MANDATORY RULES:
        1. ALWAYS call the retrieve_context tool.
        2. Answer ONLY from the retrieved context.
        3. If context is "NO_RELEVANT_CONTEXT", respond exactly:
        "I don't know based on the provided document."
        4. DO NOT use outside knowledge.
        5. DO NOT add extra explanation.
        6. DO NOT guess or infer.

        Answer Style:
        - Keep answers short and precise.
        - Use bullet points if needed.
        - Use exact wording from context when possible.
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

            # 🔥 STEP 1: Delete old folder (and all PDFs inside it)
            if os.path.exists(path):
                shutil.rmtree(path)

            # 🔥 STEP 2: Create fresh empty folder
            os.makedirs(path)

            # 🔥 STEP 3: Save only newly uploaded PDFs
            for file in uploaded:
                file_path = os.path.join(path, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())

            # 🔥 STEP 4: Reset chat + vector DB (IMPORTANT)
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.session_state.agent = None

            # 🔥 STEP 5: Process documents
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
        query_lower = query.lower().strip()

        greetings = ["hi", "hello", "hey", "good morning", "good evening"]
        thanks = ["thanks", "thank you"]
        bye = ["bye", "goodbye", "see you"]

        # -------- INTENT HANDLING --------
        if any(greet in query_lower for greet in greetings):
            answer = "Hello! 👋 Ask me anything about the uploaded document."

        elif "help" in query_lower or "what can you do" in query_lower:
            answer = "I can answer questions based on the uploaded PDF document."

        elif "who are you" in query_lower or "what are you" in query_lower:
            answer = "I am an AI assistant that answers questions from the uploaded PDF."

        elif any(t in query_lower for t in thanks):
            answer = "You're welcome! 😊"

        elif any(b in query_lower for b in bye):
            answer = "Goodbye! 👋"
        else:
            with st.spinner("Thinking..."):
                response = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": query}]},
                    {"configurable": {"thread_id": 1}}
                )
                answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai","content":answer})