from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agents import SQLDatabaseToolkit

db= SQLDatabase.from_uri("sqlite:///my_task.db")
db.run("""
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT,
    description TEXT,
    status TEXT CHECK(status IN ('pending', 'in_progress', 'completed')) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")
print("Database initialized and table created.  You can now add tasks to the 'tasks' table.")