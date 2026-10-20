from langchain.tools import tool
from retriver.Dense.generator import generate_answer
from pathlib import Path
# import mysql.connector


# @tool(description="Execute a SQL query on the database")
# def mysql_tool(query:str)->str:
#     try:
#         connection = mysql.connector.connect(
#             host="localhost",
#             user="root",
#             password="Sana$More!LikeOcean",
#             database="company"
#         )        
#         cursor=connection.cursor()
#         cursor.execute(query)
#         row=cursor.fetchall()
#         connection.close()

#         return str(row)
#     except Exception as e:
#         return f"Error:{e}"
    

@tool
def rag_tool(question: str) -> str:
    """
    use the rag tool whenever question is asked and if there is
    answer then print it if it says could not find then use 
    llm to answer make sure u do not answer first first at any 
    condition use rag tool.
    
    """
    print("====rag tool used+++")
    return generate_answer(question)
    
