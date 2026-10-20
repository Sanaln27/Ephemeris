import os
from dotenv import load_dotenv
from  langchain_chroma import Chroma
from model.config import embeddings
from .store_db import get_vector
vectorstore=get_vector()

retriver=vectorstore.as_retriever(search_kwargs={"k":3})
def retrive_documents(query:str):
    chunks=retriver.invoke(query)

    for i ,doc in enumerate(chunks,start=1):
        print(f"chunks{i}")
        print(doc.page_content)
        print("metadate",doc.metadata)
    return chunks

