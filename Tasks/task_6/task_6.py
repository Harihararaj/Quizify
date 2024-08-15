import sys
import os
import streamlit as st
sys.path.append(os.path.abspath('../../'))
from Tasks.task_3.task_3 import DocumentProcessor
from Tasks.task_4.task_4 import EmbeddingClient
from Tasks.task_5.task_5 import ChromaCollectionCreator


if __name__ == "__main__":
    st.header("Quizzify")

    # Configuration for EmbeddingClient
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "sample-mission-quizify-432319",
        "location": "us-central1"
    }
    
    screen = st.empty() # Screen 1, ingest documents
    with screen.container():
        processor = DocumentProcessor() # Initialize from Task 3
        processor.ingest_documents()

        embed_client = EmbeddingClient(**embed_config) # Initialize from Task 4
    
        chroma_creator = ChromaCollectionCreator(processor, embed_client)
        

        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
            
            topic_input = st.text_input(label = "",placeholder="Enter the topic")
            questions = st.select_slider("Select number of questions", options = [1,2,3,4,5,6,7,8,9,10], value = 5)
            
            document = None
            
            submitted = st.form_submit_button("Generate a Quiz!")
            if submitted:
                if not topic_input.strip():
                    st.error("Text content cannot be empty. Please provide some text.")
                else:
                    chroma_creator.create_chroma_collection()
                    document = chroma_creator.query_chroma_collection(topic_input)
    if document:
        screen.empty() # Screen 2
        with st.container():
            st.header("Query Chroma for Topic, top Document: ")
            st.write(document)