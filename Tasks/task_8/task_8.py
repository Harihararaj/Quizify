import streamlit as st
import os
import sys
import json
sys.path.append(os.path.abspath('../../'))
from Tasks.task_3.task_3 import DocumentProcessor
from Tasks.task_4.task_4 import EmbeddingClient
from Tasks.task_5.task_5 import ChromaCollectionCreator

from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI

class QuizGenerator:
    def __init__(self, topic=None, num_questions=1, vectorstore=None):
        """
        """
        if not topic:
            self.topic = "General Knowledge"
        else:
            self.topic = topic

        if num_questions > 10:
            raise ValueError("Number of questions cannot exceed 10.")
        self.num_questions = num_questions

        self.vectorstore = vectorstore
        self.llm = None
        self.question_bank = [] # Initialize the question bank to store questions
        self.system_template = """
            You are a subject matter expert on the topic: {topic}
            
            Follow the instructions to create a quiz question:
            1. Generate a question based on the topic provided and context as key "question"
            2. Provide 4 multiple choice answers to the question as a list of key-value pairs "choices"
            3. Provide the correct answer for the question from the list of answers as key "answer"
            4. Provide an explanation as to why the answer is correct as key "explanation"
            
            You must respond as a JSON object with the following structure:
            {{
                "question": "<question>",
                "choices": [
                    {{"key": "A", "value": "<choice>"}},
                    {{"key": "B", "value": "<choice>"}},
                    {{"key": "C", "value": "<choice>"}},
                    {{"key": "D", "value": "<choice>"}}
                ],
                "answer": "<answer key from choices list>",
                "explanation": "<explanation as to why the answer is correct>"
            }}
            
            Context: {context}
            """
    
    def init_llm(self):
        """
        """
        self.llm = VertexAI(
            model_name = "gemini-pro",
            temperature = 0.8, # Increased for less deterministic questions 
            max_output_tokens = 500
        )

    def generate_question_with_vectorstore(self):
        """
        """
        if not self.llm:
            self.init_llm()
        if not self.vectorstore:
            raise ValueError("Vectorstore not provided.")
        
        from langchain_core.runnables import RunnablePassthrough, RunnableParallel

        # Enable a Retriever
        retriever = self.vectorstore.get_retriever()
        
        # Use the system template to create a PromptTemplate
        prompt = PromptTemplate.from_template(self.system_template)
        
        # RunnableParallel allows Retriever to get relevant documents
        # RunnablePassthrough allows chain.invoke to send self.topic to LLM
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "topic": RunnablePassthrough()}
        )
        # Create a chain with the Retriever, PromptTemplate, and LLM
        chain = setup_and_retrieval | prompt | self.llm 
        # Invoke the chain with the topic as input
        response = chain.invoke(self.topic)
        return response

    def generate_quiz(self) -> list:
        """
        """
        self.question_bank = []
        for _ in range(self.num_questions):
            question_str = self.generate_question_with_vectorstore()
            try:
                question_str = question_str[question_str.find('{') : question_str.rfind('}')+1]
                question_str.replace('\n', '').replace('\r', '')
                question = json.loads(question_str)
            except Exception as e:
                print("Failed to decode question JSON.",e)
                print(question_str)
                continue
            if self.validate_question(question):
                print("Successfully generated unique question")
                self.question_bank.append(question)
            else:
                print("Duplicate or invalid question detected.")

        return self.question_bank
    
    def validate_question(self, question: dict) -> bool:
        """
        """
        if "question" not in question:
            return False
        for q in self.question_bank:
            if q["question"] == question["question"]:
                return False
        return True