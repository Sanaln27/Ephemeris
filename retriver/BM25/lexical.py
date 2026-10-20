from langchain_community.retrievers import BM25Retriever
from .chunks import chunk
import re

def preprocess(text):
    text=text.lower()
    text=re.sub(r"[^a-zA-Z0-9\n]"," ",text)
    stopword={
                "what", "is", "the", "a", "an", "of", "to",
        "and", "in", "for", "on", "are", "was", "how","this"

    }

    return[
        word
        for word in text.split()
        if word  not in stopword 
    ]

retriver=BM25Retriever.from_documents(
    chunk,
    k=3,
    preprocess_func=preprocess
)



