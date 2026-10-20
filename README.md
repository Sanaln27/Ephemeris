# Agentic RAG System using LangChain

This project is an **Agentic Retrieval-Augmented Generation (RAG) System** built using **LangChain**. The main purpose of this project is to perform **semantic search** over documents and generate accurate answers based on the retrieved information. Along with document retrieval, the system can also query a MySQL database using tool calling.


# How the System Works

The first step of the project is document ingestion. The user places all the required documents inside the **data** folder. The system supports multiple file formats such as:

.pdf,.txt,.csv,.dox, Markdown (.md).

Instead of creating a separate loader for every file type, the project follows a modular approach using different helper functions.

- **get_files()** – Goes inside the given folder and fetches all the available files.
- **check_files()** – Checks whether a particular file type is supported by the system.
- **get_loader()** – Selects the appropriate LangChain document loader depending on the file type.
- **load_documents()** – Reads the contents of all supported files and converts them into LangChain **Document** objects. These document objects temporarily store the extracted text, making it easier for further AI processing.

# Chunking and Embedding

Once all the text has been extracted from the documents, the next step is **chunking**.

Large documents cannot be processed efficiently in one go, so they are divided into smaller overlapping chunks using LangChain's **Recursive Character Text Splitter**.

Each chunk is then converted into an **embedding** using OpenAI Embeddings. These embeddings capture the semantic meaning of the text rather than just the keywords.

Finally, all the generated embeddings are stored inside **ChromaDB**, which acts as the vector database for this project.

# Complete RAG Pipeline

```
Documents (data/)
        │
        ▼
load_documents()
(document_loader.py)
        │
        ├── get_all_files()
        ├── is_supported()
        ├── get_loader()
        ▼
LangChain Document Objects
        │
        ▼
chunk_documents()
(chunking.py)
        │
        ▼
OpenAI Embeddings
(config.py)
        │
        ▼
ChromaDB
(Vector Database)

```

# Retrieval-Augmented Generation (RAG)

When a user asks a question, the question is first converted into an embedding.

The system then performs **semantic similarity search** on ChromaDB to retrieve the most relevant document chunks. These retrieved chunks are passed as context to the Large Language Model (LLM), which generates an answer based on the retrieved information.

This completes the RAG pipeline.

```
Documents
      │
      ▼
Document Loader
      │
      ▼
Chunking
      │
      ▼
Embeddings
      │
      ▼
ChromaDB
      │
      ▼
Retriever
      │
      ▼
LLM
      │
      ▼
Answer
```

---

# Tool Calling

After building the RAG pipeline, it is integrated as a **LangChain Tool**.

The project currently contains two tools:

### 1. RAG Tool

This tool retrieves relevant information from the documents stored in ChromaDB and answers document-related questions using semantic search.

### 2. MySQL Tool

This tool executes SQL queries on the connected MySQL database and retrieves structured information whenever required.

The LangChain Agent automatically decides which tool should be used depending on the user's query.


# Prompt Engineering

The system prompt is stored separately inside the **prompts** folder.

Keeping prompts in a separate file makes the project easier to maintain because prompts usually require frequent modifications during development. Instead of changing the main application code, only the prompt file needs to be updated.


# Main Application

The **app.py** file acts as the entry point of the project.

It initializes:

- The Large Language Model (LLM)
- The available tools
- The LangChain Agent

Whenever the user asks a question, the agent decides whether the query should be answered using:

- The **RAG Tool**
- The **MySQL Tool**
- Or directly by the language model.

If the required information is found in the documents, the RAG Tool performs semantic retrieval and generates an answer using the retrieved context.

If the question is unrelated to the indexed documents or database, the language model answers using its own general knowledge.

```
User Question
        │
        ▼
app.py
        │
        ▼
LangChain Agent
        │
 ┌──────┴─────────┐
 ▼                ▼
rag_tool()   mysql_tool()
        
```

# Technologies Used

- Python
- LangChain
- OpenAI
- ChromaDB
- MySQL
- python-dotenv


