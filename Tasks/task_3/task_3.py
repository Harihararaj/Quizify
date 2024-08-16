import streamlit as st
from io import StringIO
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile
import uuid
class DocumentProcessor:
    """
    This class encapsulates the functionality for processing uploaded PDF documents using Streamlit
    and Langchain's PyPDFLoader. It provides a method to render a file uploader widget, process the
    uploaded PDF files, extract their pages, and display the total number of pages extracted.
    """
    def __init__(self):
        self.pages = []  # List to keep track of pages from all documents
    
    def ingest_documents(self):
        uploaded_files = st.file_uploader(
            "Choose a pdf file", 
            accept_multiple_files=True,
            type = ['pdf'],
            key='pdfs'
        )

        if uploaded_files is not None:
            for uploaded_file in uploaded_files:

                unique_id = uuid.uuid4().hex
                original_name, file_extension = os.path.splitext(uploaded_file.name)
                temp_file_name = f"{original_name}_{unique_id}{file_extension}"
                temp_file_path = os.path.join(tempfile.gettempdir(), temp_file_name)

                # Write the uploaded PDF to a temporary file
                with open(temp_file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()

                for document in documents:
                    self.pages.append(document)
                
                # Clean up by deleting the temporary file.
                os.unlink(temp_file_path)
            
            # Display the total number of pages processed.
            st.write(f"Total pages processed: {len(self.pages)}")