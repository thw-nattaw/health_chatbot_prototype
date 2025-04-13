from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Load LLaMA model once for reuse
llm = Ollama(mmodel="llama3.1")

# Prompt for generating SOAP summary from conversation
soap_template = """
You are a medical assistant tasked with summarizing a patient interview into SOAP note format.

Given the following conversation between the patient and AI assistant, generate a structured SOAP note.

Format:
S: Subjective - the patient's symptoms and concerns
O: Objective - any observable facts or measurements (leave blank if not available)
A: Assessment - possible diagnoses (based on conversation context)
P: Plan - recommended next steps or information to gather (without performing actual medical intervention)

Conversation:
{conversation_history}

SOAP Summary:
"""

def summarize_conversation(conversation_history: str) -> str:
    """Generate a SOAP summary from the patient-AI conversation"""
    prompt = ChatPromptTemplate.from_template(soap_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"conversation_history": conversation_history})
