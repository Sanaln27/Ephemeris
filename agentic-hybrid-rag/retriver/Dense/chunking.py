import hashlib
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from .dc_loader import load_documents
from .store_db import get_vector

load_dotenv()


def build_vector_database():
    documents = load_documents("data1")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " "],
    )

    chunks = splitter.split_documents(documents)

    vectorstore = get_vector()
    existing_ids = set(vectorstore.get()["ids"])

    new_chunks = []
    new_ids = []
    chunk_numbers = {}

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")

        chunk_numbers[source] = chunk_numbers.get(source, 0) + 1
        chunk_number = chunk_numbers[source]

        id_input = f"{source}\0{chunk_number}\0{chunk.page_content}"
        chunk_id = hashlib.sha256(id_input.encode("utf-8")).hexdigest()

        if chunk_id in existing_ids:
            print(f"Already exists, skipping ID: {chunk_id}")
            continue

        chunk.metadata["indexed_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        new_chunks.append(chunk)
        new_ids.append(chunk_id)
        print(f"New chunk ID: {chunk_id}")

    if not new_chunks:
        print("No new chunks to index.")
        return

    vectorstore.add_documents(new_chunks, ids=new_ids)
    print(f"Added {len(new_chunks)} new chunks.")