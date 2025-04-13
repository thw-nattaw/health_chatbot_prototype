from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = Ollama(model="llama3.1:latest")

# Prompt template for the interview phase (used in the chat UI)
interview_template = """
You are an AI-powered virtual medical assistant designed to interview patients before they meet a physician in Japanese Language.
Your task is to
1. Gather detailed information about the symptom.
2. Ask follow-up questions to ensure the information is clear and complete. Only 1 question per turn.

Current conversation:
{conversation_history}

Only ask in Japanese. No translation in English is required.
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
# Prompt for generating summary from conversation
summary_template = """
You are a Japanese medical assistant tasked with summarizing a patient interview.

Given the following conversation between the patient and AI assistant, generate a History of Present Illness in the medical records、in Japanese Language.

Conversation:
{conversation_history}

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

def summarize_conversation(conversation_history: str) -> str:
    """Generate a summary from the patient-AI conversation"""
    prompt = ChatPromptTemplate.from_template(summary_template )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"conversation_history": conversation_history})