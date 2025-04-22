from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.output_parsers import StrOutputParser
import time

llm = OllamaLLM(model="llama3.3")

# Prompt template for the interview phase (used in the chat UI)
interview_template = """You are an AI-powered virtual medical assistant designed to interview patients before they meet a physician in Japanese Language.
1. Clarify the patient's chief complaint and gather detailed information about the symptom.
2. Ask follow-up questions to ensure the information is clear and complete.
3. Ask only one question per turn.
4. Try to ask about details of symptoms that patient provided first.
5. Try to ask question that beneficial to the differential diagnosis

Patient Interview Transcript:
{conversation_history}
"""

# Prompt for generating summary from conversation
summary_template = """
You are a Japanese medical assistant tasked with summarizing a patient interview.

Given the following conversation between the patient and AI assistant, generate a History of Present Illness in the medical records、in Japanese Language.

Conversation:
{conversation_history}
"""

def get_interview_response(user_input, conversation_history):
    """Run the interview phase with the current conversation context"""
    start_time = time.time()
    prompt= ChatPromptTemplate.from_template(interview_template)
    chain = prompt | llm 
    output = chain.invoke({"conversation_history": conversation_history})
    end_time = time.time() 
    execution_time = end_time - start_time
    print(f"Time spent: {execution_time:.4f} seconds")
    return output

def summarize_conversation(conversation_history: str) -> str:
    """Generate a summary from the patient-AI conversation"""
    prompt = ChatPromptTemplate.from_template(summary_template )
    chain = prompt | llm 
    return chain.invoke({"conversation_history": conversation_history})
