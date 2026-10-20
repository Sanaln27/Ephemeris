import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage,AIMessage
from tools.tools import rag_tool
from model.config import llm
from memory.conversation import get_hstry,add_msg
from memory.memory_agent import memory_agent

load_dotenv()

# connection = mysql.connector.connect(
#     host=os.getenv("MYSQL_HOST"),
#     user=os.getenv("MYSQL_USER"),
#     password=os.getenv("MYSQL_PASSWORD"),
#     database=os.getenv("MYSQL_DATABASE")
# )


tools=[
    # mysql_tool,
    rag_tool
    ]

agent=create_agent(
    model=llm,
    tools=tools
)

while True:
    question=input("\nmeee:\n")
    if question.lower()=="exit":
        print("boyee boyeee")
        break 
    history=get_hstry()

    response = agent.invoke({
        "messages":history+[{
         
            "role":"user",
            "content":question

        }

        ]
    })

    answer=response["messages"][-1].content

    memory_agent(llm,question,answer)


    print(answer)