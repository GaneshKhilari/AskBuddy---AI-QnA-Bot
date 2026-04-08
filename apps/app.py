#from dotenv import load_dotenv
import os
#load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# while True:
#     query = input("User: ")
#     if(query.lower()) in ["quit","bye","exit"]:
#         print("GoodBye👋")
#         break
#     result =llm.invoke(query)
#     print("AI: ", result.content)

import streamlit as st
st.title("🧠 AskBuddy - AI QnA Bot")        #title idea : "Curious? Just ask!"
st.markdown("My QnA Bot with LangChain and Googl Gemini !")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask anything ?")
if query :
    st.session_state.messages.append({"role":"user", "content":query})
    st.chat_message("user").markdown(query)
    # res = llm.invoke(query)
    # st.chat_message("ai").markdown(res.content)
    # st.session_state.messages.append({"role":"ai", "content":res.content})
    with st.spinner("Thinking... 🤔"):
        try:
            res = llm.invoke(query)
            response = res.content
        except Exception as e:
            response = f"❌ Error: {str(e)}"

    st.chat_message("ai").markdown(response)
    st.session_state.messages.append({"role":"ai", "content":response})
