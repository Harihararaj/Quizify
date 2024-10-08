import streamlit as st
import os
import sys
import json
sys.path.append(os.path.abspath('../../'))
from Tasks.task_3.task_3 import DocumentProcessor
from Tasks.task_4.task_4 import EmbeddingClient
from Tasks.task_5.task_5 import ChromaCollectionCreator
from Tasks.task_8.task_8 import QuizGenerator

class QuizManager:
    def __init__(self, questions: list):
        """
        """
        self.questions = questions
        self.total_questions = len(self.questions)

    def get_question_at_index(self, index: int):
        """
        """
        valid_index = index % self.total_questions
        return self.questions[valid_index]
    
    def next_question_index(self, direction=1):
        """
        """
        if 'question_index' not in st.session_state:
            st.session_state['question_index'] = 0
        st.session_state['question_index'] = (st.session_state['question_index'] + direction) % self.total_questions

# Test Generating the Quiz
if __name__ == "__main__":
    
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "sample-mission-quizify-432319",
        "location": "us-central1"
    }
    
    screen = st.empty()
    if 'question_bank' not in st.session_state or len(st.session_state.question_bank) ==0 :
        with screen.container():
            st.header("Quiz Builder")
            processor = DocumentProcessor()
            processor.ingest_documents()
        
            embed_client = EmbeddingClient(**embed_config) 
        
            chroma_creator = ChromaCollectionCreator(processor, embed_client)
        
            question = None
            question_bank = None
        
            with st.form("Load Data to Chroma"):
                st.subheader("Quiz Builder")
                st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
                
                topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
                questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
                
                submitted = st.form_submit_button("Submit")
                if submitted:
                    chroma_creator.create_chroma_collection()
                    
                    st.write(topic_input)
                    
                    # Test the Quiz Generator
                    generator = QuizGenerator(topic_input, questions, chroma_creator)
                    st.session_state.question_bank = generator.generate_quiz()
                    st.rerun()
    else:
        if 'question_index' in st.session_state:
            question_index = st.session_state.question_index
        else:
            question_index = 0
            st.session_state.question_index = 0
        screen.empty()
        with st.container():
            st.header("Generated Quiz Question: ")
            quiz_manager = QuizManager(st.session_state.question_bank)
            # Format the question and display
            index_question = quiz_manager.get_question_at_index(question_index)
            choices = []
            for choice in index_question['choices']:
                key = choice['key']
                value = choice['value']
                choices.append(f"{key}) {value}")
            
            st.write(index_question['question'])
            
            answer = st.radio(
                'Choose the correct answer',
                choices
            )
            answer_submitted = st.button("Submit Answer")
            next_question = st.button("Next Question")
            previous_question = st.button("Previous Question")
            
            if answer_submitted:
                correct_answer_key = index_question['answer']
                if answer.startswith(correct_answer_key): # Check if answer is correct
                    st.success("Correct!")
                    st.write()
                else:
                    st.error("Incorrect!")

            if next_question:
                quiz_manager.next_question_index(direction=1)
                st.rerun()

            if previous_question:
                quiz_manager.next_question_index(direction=-1)
                st.rerun()