from langchain_chroma import Chroma
from model.config import embeddings

#get_vector() returns a persistent Chroma vector database configured with an embedding model to store and search text by meaning.
def get_vector():
    return Chroma(
    persist_directory="Chroma_db",
    embedding_function=embeddings,
)