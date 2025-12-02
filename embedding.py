import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

INDEX_FOLDER_NAME = "faiss_index"
INDEX_FILE_NAME = "qiskit_docs_index"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def get_all_docs_from_qiskit_data():
    """
    Get all qiskit documentation from Qiskit data.
    """
    all_docs = []
    for filename in os.listdir("./qiskit-data/1.4"):
        file_path = os.path.join("./qiskit-data/1.4", filename)
        if filename.endswith(('.txt', '.py', '.ini', 'mdx')):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                all_docs.append(Document(
                    page_content=text,
                    metadata={"source": filename, "type": "code" if filename.endswith('.py') else "text"}
                ))
    return all_docs

def chunk_docs(documents):
    """
    Split documents into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    splits = text_splitter.split_documents(all_docs)
    for doc in splits:
        doc.page_content = doc.page_content.encode("utf-8", "ignore").decode("utf-8")

    return splits


# --- RAG/Embedding Section ---
if __name__ == "__main__":
    load_dotenv()
    if not os.environ["OPENAI_API_KEY"]:
        print("OpenAI API key is required for embedding generation. Please enter your OpenAI API key in .env file.")

    embeddings = OpenAIEmbeddings()
    all_docs = get_all_docs_from_qiskit_data()
    splits = chunk_docs(all_docs)
    
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(INDEX_FOLDER_NAME, INDEX_FILE_NAME)
