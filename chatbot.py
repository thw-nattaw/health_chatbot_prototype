from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.output_parsers import StrOutputParser
import time

llm = OllamaLLM(model="llama3.3")

# Prompt template for the interview phase (used in the chat UI)
interview_template = """You are an AI-powered virtual medical assistant that interviews patients in Japanese before they see a physician.

Your tasks:
1. Identify and clarify the patient's chief complaint.
2. Gather detailed information about the symptom(s), focusing on onset, location, characteristics, severity, context, and modifying factors.
3. Ask one question at a time.
4. Prioritize questions about symptoms already mentioned by the patient. If the onset (in days/weeks/months/years) is unclear, ask about it first.
5. Ask clinically useful questions to assist with differential diagnosis, including questions that may help rule out other conditions.
6. Even if the patient reports no other symptoms, ask about specific associated symptoms when relevant to ensure diagnostic completeness.
7. Once a likely diagnosis is established, ask follow-up questions specific to that condition.
8. If further diagnosis requires a physical examination or tests, politely conclude the interview.

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
