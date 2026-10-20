from model.config import llm
from .retriver import retrive_documents
from fhfh.memory.prompts.rag_prompt import rag_prompt

def generate_answer(question:str):
    documents=retrive_documents(question)
    content="\n\n".join(
        f"Source file: {doc.metadata.get('source', 'unknown')}\n"
        f"Indexed at: {doc.metadata.get('indexed_at', 'unknown')}\n"
        f"Content:\n{doc.page_content}"
        # f"Source file: {doc.metadata.get('source', 'unknown')}\n"
        # f"Content:\n{doc.page_content}"
        for doc in documents
    )

    prompt=rag_prompt.invoke({ 
        "context":content,
        "question":question
    })
    response=llm.invoke(prompt)
    return response.content



