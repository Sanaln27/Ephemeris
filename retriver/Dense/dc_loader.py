from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)


# Supported file types and their loaders
LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
    ".xlsx": UnstructuredMarkdownLoader,
}


def get_files(folder_path: str):
    """Returns all files inside the folder."""

    folder = Path(folder_path)

    return [
        file
        for file in folder.iterdir()
        if file.is_file()
    ]


def check_file(file):
    """Checks whether the file type is supported."""

    return file.suffix.lower() in LOADERS


def get_loader(file):
    """Returns the correct loader for the file."""

    return LOADERS[file.suffix.lower()]


def load_documents(folder_path: str):
    """
    Reads every supported file inside the folder
    and returns a list of LangChain Document objects.
    """

    documents = []

    files = get_files(folder_path)

    for file in files:

        if not check_file(file):
            print(f"Skipping: {file.name}")
            continue

        print(f"Loading: {file.name}")

        loader = get_loader(file)

        docs = loader(str(file)).load()

        documents.extend(docs)

    return documents