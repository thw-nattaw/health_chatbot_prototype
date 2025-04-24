from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
import re

llm = OllamaLLM(model="llama3.3")

# Prompt template for the interview phase (used in the chat UI)
interview_template = """You are an AI-powered virtual medical assistant that interviews patients in Japanese before they see a physician.

Your highest priority:
- Your goal is to help identify the **most likely diagnosis** or **diagnostic possibilities**, not just one explanation. Avoid anchoring on the first hypothesis. Stay flexible and broad in your differential reasoning.

Your diagnostic tasks:
1. Identify and clarify the patient's chief complaint.
2. Collect detailed symptom information (onset, location, characteristics, severity, context, modifying factors).
3. If the onset (in days/weeks/months/years) is unclear, ask about it first.
4. Ask only one question at a time.

Diagnostic strategy:
5. After exploring the main symptom, screen for relevant associated symptoms from **multiple organ systems** (e.g., cardiovascular, respiratory, hematologic, neurologic, endocrine, gynecologic) — especially those that may indicate serious or common etiologies.
6. Explore potential causes systematically
7. Only after sufficient symptom and etiology screening, ask about medical history, treatment, smoking, alcohol, and family history.

Diagnosis refinement:
8. Once you suspect a likely diagnosis, ask focused follow-up questions to support or challenge that hypothesis.
9. Conclude the interview politely when the next step clearly requires physical exam or tests (e.g., bloodwork, imaging).


**Important reminders**:
- Do not stop at one plausible cause — collect enough information to confirm or rule out multiple potential diagnoses.
- Always consider common, treatable, or serious conditions that may present subtly.
- Please do not include any thought processes other than the question (例：「〜が考えられます」「〜の可能性があります」など）
- Do not ask for information already answered.

Patient Information:
- 年齢: {age}歳
- 性別: {gender}

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
ALLOWED_ENGLISH_TERMS = {
    "COPD", "NSAIDs", "CT", "MRI"
}

def is_valid_japanese_question(output: str) -> bool:
    # Extract ALL non-Japanese character tokens (including Latin, Cyrillic, etc.)
    non_japanese_words = re.findall(r'[^\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+', output)
    
    # Now extract actual alphanumeric "words" from those segments
    foreign_words = []
    for segment in non_japanese_words:
        words = re.findall(r'\b[A-Za-z0-9]+\b', segment)
        foreign_words.extend(words)

    # Filter out allowed English words
    non_whitelisted = [w for w in foreign_words if w not in ALLOWED_ENGLISH_TERMS]

    if non_whitelisted:
        print(f"[WARNING] 不許可の英語／非日本語表現が含まれています: {non_whitelisted}")
        return False

    return True


def get_interview_response(conversation_history, age=None, gender=None, max_retries=3):
    start_time = time.time()
    prompt = ChatPromptTemplate.from_template(interview_template)
    chain = prompt | llm

    retries = 0
    question_output = ""
    while retries < max_retries:
        question_output = chain.invoke({
            "conversation_history": conversation_history,
            "age": age,
            "gender": gender
        })
        if is_valid_japanese_question(question_output):
            break
        retries += 1

    if retries == max_retries:
        question_output = "[システムエラー] 有効な質問を生成できませんでした。"

    print(f"Time spent: {time.time() - start_time:.4f} seconds")
    return question_output



def summarize_conversation(conversation_history: str) -> str:
    """Generate a summary from the patient-AI conversation"""
    prompt = ChatPromptTemplate.from_template(summary_template )
    chain = prompt | llm 
    return chain.invoke({"conversation_history": conversation_history})
