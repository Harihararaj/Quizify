from langchain_google_vertexai import VertexAIEmbeddings

class EmbeddingClient:
    """
    """
    def __init__(self, model_name, project, location):
        self.client = VertexAIEmbeddings(
            model_name = model_name,
            project = project,
            location = location
        )
        
    def embed_query(self, query):
        """
        """
        vectors = self.client.embed_query(query)
        return vectors
    
    def embed_documents(self, documents):
        """
        """
        try:
            return self.client.embed_documents(documents)
        except AttributeError:
            print("Method embed_documents not defined for the client.")
            return None