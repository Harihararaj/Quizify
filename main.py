
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
        st.session_state['question_index'] = (st.session_state['question_index'] + direction)

# Test Generating the Quiz
if __name__ == "__main__":

    def options(option):
        if(option == 'A'):
            return 1
        elif(option == 'B'):
            return 2
        elif(option == 'C'):
            return 3
        else:
            return 4
    
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
                st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
                topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
                questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
                submitted = st.form_submit_button("Generate")
                if submitted:
                    chroma_creator.create_chroma_collection()
                    generator = QuizGenerator(topic_input, questions, chroma_creator)
                    st.session_state.question_bank = generator.generate_quiz()
                    st.rerun()
    elif 'question_index' in st.session_state and st.session_state.question_index == len(st.session_state.question_bank):
        screen.empty()

        with st.container():
            with st.form("Test Result"):
                st.subheader("Test Result...")
                total_score = len(st.session_state.question_bank)
                score = st.session_state.score
                if(score/total_score >= 0.5):
                    st.success(f'**Score :** {score}/{total_score}')
                else:
                    st.error(f'**Score :** {score}/{total_score}')
                end_test = st.form_submit_button("End Test")
                if end_test:
                    st.session_state.clear()
                    st.rerun()


    else:
        if 'question_index' in st.session_state:
            if(st.session_state.question_index <= 0):
                st.session_state.question_index =0
            question_index = st.session_state.question_index
        else:
            question_index = 0
            st.session_state.question_index = 0
            st.session_state.score = 0
        if "answered_correct" in st.session_state.question_bank[question_index]:
            index=st.session_state.question_bank[question_index]["option"] - 1
            disabled = True
        else:
            index=None
            disabled = False
        screen.empty()
        with st.container():
            with st.form("Multiple Choice Question"):
                st.subheader("Generated Quiz Question: ")
                quiz_manager = QuizManager(st.session_state.question_bank)
                # Format the question and display
                index_question = quiz_manager.get_question_at_index(question_index)
                choices = []
                for choice in index_question['choices']:
                    key = choice['key']
                    value = choice['value']
                    choices.append(f"{key}) {value}")
                
                st.markdown(f"**{question_index+1}){index_question['question']}**")
                answer = st.radio(
                    'Choose the correct answer',
                    choices,
                    index = index,
                    disabled = disabled
                )
                answer_submitted = st.form_submit_button("Submit Answer", disabled=disabled)
                if answer_submitted:
                    correct_answer_key = index_question['answer']
                    st.session_state.question_bank[question_index]["option"] = options(answer[0])
                    if answer.startswith(correct_answer_key): # Check if answer is correct
                        st.session_state.question_bank[question_index]["answered_correct"] = True
                        st.session_state.score += 1
                        st.rerun()
                    else:
                        st.session_state.question_bank[question_index]["answered_correct"] = False
                        st.write(answer[0])
                        st.rerun()
                if "answered_correct" in st.session_state.question_bank[question_index] and st.session_state.question_bank[question_index]["answered_correct"] == True:
                    st.success(f'Correct!')
                    st.markdown(f'**Explanation:** {index_question["explanation"]}')
                elif "answered_correct" in st.session_state.question_bank[question_index] and st.session_state.question_bank[question_index]["answered_correct"] == False:
                    st.error("Incorrect!")

            with st.container():
                col1, col2, col3 = st.columns([2, 6.75, 1.25])
                with col1:
                    prev_button = st.button("⬅️ Previous", key="prev_button", on_click=lambda: quiz_manager.next_question_index(direction=-1))
                with col3:
                    next_button = st.button("Next ➡️", key="next_button", on_click=lambda: quiz_manager.next_question_index(direction=1))