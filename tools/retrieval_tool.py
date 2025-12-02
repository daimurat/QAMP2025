from langchain_core.vectorstores import VectorStore

class RetrievalTool:
    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    def retrieve(self, query, top_k=5):
        docs = self.vs.similarity_search(query, k=top_k)
        return "\n\n".join([doc.page_content for doc in docs])