import sys
import os
import streamlit as st
sys.path.append(os.path.abspath('../../'))
from Tasks.task_3.task_3 import DocumentProcessor
from Tasks.task_4.task_4 import EmbeddingClient


# Import Task libraries
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

class ChromaCollectionCreator:
    def __init__(self, processor, embed_model):
        """
        """
        self.processor = processor      # This will hold the DocumentProcessor from Task 3
        self.embed_model = embed_model  # This will hold the EmbeddingClient from Task 4
        self.db = None                  # This will hold the Chroma collection

    def create_chroma_collection(self):
        """
        """
        
        # Step 1: Check for processed documents
        if len(self.processor.pages) == 0:
            st.error("No documents found!", icon="🚨")
            return
        text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
        texts = text_splitter.split_documents(self.processor.pages)
        
        if texts is not None:
            st.success(f"Successfully split pages to {len(texts)} documents!", icon="✅")
            
        self.db = Chroma.from_documents(documents = texts, embedding = self.embed_model)
        
        if self.db:
            st.success("Successfully created Chroma Collection!", icon="✅")
        else:
            st.error("Failed to create Chroma Collection!", icon="🚨")
    def query_chroma_collection(self, query) -> Document:
        """
        """
        if self.db:
            docs = self.db.similarity_search(query,k=2)
            if docs:
                return docs[0]
            else:
                st.error("No matching documents found!", icon="🚨")
        else:
            st.error("Chroma Collection has not been created!", icon="🚨")
    
    def get_retriever(self):
        return self.db.as_retriever()
