from memory.memory_agent import memory_agent
from model.config import llm

memory_agent(
    llm=llm,
    user_imput="My name is Sana",
    assistant_response="Nice to meet you."
)

print("Memory Agent Executed Successfully!")