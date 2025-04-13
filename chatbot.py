from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Load the Ollama LLaMA 3.1 model
llm = Ollama(model="llama3.1:latest")

# Prompt template for the interview phase (used in the chat UI)
interview_template = """
You are an AI-powered virtual medical assistant designed to interview patients before they meet a physician.
1. Clarify the patient's chief complaint and gather detailed information about the symptom.
2. Ask follow-up questions to ensure the information is clear and complete.
3. Do NOT suggest a diagnosis or mention possible conditions during this phase.
4. End the interview when enough information is collected.

Current conversation:
{conversation_history}

Patient: {user_input}
AI Assistant:
"""

# Prompt template for silent diagnosis (internal reasoning only)
diagnosis_template = """
You are a clinical assistant tasked with analyzing a patient interview to generate a list of possible diagnoses.
1. Analyze the collected patient information and provide a list of likely diagnoses.
2. Consider common and rare conditions that match the symptoms and history.
3. Recommend what additional questions should be asked to confirm or rule out the most likely diagnosis.

Patient interview:
{conversation_history}

Possible diagnoses and recommended follow-up questions:
"""

def get_interview_response(user_input, conversation_history):
    """Run the interview phase with the current conversation context"""
    prompt = ChatPromptTemplate.from_template(interview_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"conversation_history": conversation_history, "user_input": user_input})

def get_diagnosis(conversation_history):
    """Run the diagnosis phase silently (not shown to user)"""
    prompt = ChatPromptTemplate.from_template(diagnosis_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"conversation_history": conversation_history})
