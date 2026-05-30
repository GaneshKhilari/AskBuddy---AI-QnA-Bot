from dotenv import load_dotenv
load_dotenv(dotenv_path="C:\\GenAI\\.env")

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

llm = ChatGroq(model ="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()
memory=MemorySaver()

agent = create_agent(
        model =llm,
        tools =[search.run],
        checkpointer=memory,
        system_prompt="Your are a agent and can search for any question on goole."
    )

while True:
    query = input("User: ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": "1"}}
        )
    
    print("AI:", response["messages"][-1].content)