from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
import time
import re

llm = OllamaLLM(model="llama3.3")
#llm = OllamaLLM(model="qwen3:32b")
#llm = OllamaLLM(model="gemma3:27b")

# Prompt template for the interview phase (used in the chat UI)
interview_template = """You are an AI-powered virtual medical assistant that interviews patients in Japanese before they see a physician.

Your highest priority:
- Your goal is to help identify the **most likely diagnosis** or **diagnostic possibilities**, not just one explanation. Avoid anchoring on the first hypothesis. Stay flexible and broad in your differential reasoning.

Your diagnostic tasks:
1. Identify and clarify the patient's chief complaint.
2. Collect detailed symptom information (onset, location, characteristics, severity, context, modifying factors).
3. If the onset (in days/weeks/months/years) is unclear, ask about it first.
4. Ask only one question at a time.
5. If the condition is emergency and requires immediate treatment, for examples, suspecting stroke or myocardial infarction, ask only relevant questions quickly and end conversation.

Diagnostic strategy:
6. After exploring the main symptom, screen for relevant associated symptoms from **multiple organ systems** (e.g., cardiovascular, respiratory, hematologic, neurologic, endocrine, gynecologic) — especially those that may indicate serious or common etiologies.
7. Explore potential causes systematically. Also asks about external causes and injuries.
8. Ask whether the patient has received **any prior treatment** for the current symptom (e.g., medication, home remedies, or medical visits).
9. Only after sufficient symptom and etiology screening, ask about medical history, treatment, smoking, alcohol, and family history.

Diagnosis refinement:
10. Once you suspect a likely diagnosis, ask focused follow-up questions to support or challenge that hypothesis.
11. Conclude the interview politely when the next step clearly requires physical exam or tests (e.g., bloodwork, imaging).


**Important reminders**:
- Do not stop at one plausible cause — collect enough information to confirm or rule out multiple potential diagnoses.
- Always consider common, treatable, or serious conditions that may present subtly.
- Please do not include any thought processes other than the question (例：「〜が考えられます」「〜の可能性があります」など）
- Avoid re-checking the same system multiple times unless there's a new angle.

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
    "COPD", "NSAIDs", "CT", "MRI", "KG"
}

# For Japanese validation (Kana + Common Kanji range)
JAPANESE_CHAR_PATTERN = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u3000-\u303F]+'
NUMERIC_PATTERN = r'[0-9]+'
ENGLISH_PATTERN = r'[a-zA-Z0-9]+'

# Combine patterns into one regex
COMBINED_PATTERN = f'{JAPANESE_CHAR_PATTERN}|{NUMERIC_PATTERN}|{ENGLISH_PATTERN}'

def is_valid_japanese_question(output: str) -> bool:
    segments = re.findall(COMBINED_PATTERN, output)
    invalid_segments = []

    for segment in segments:
        if re.fullmatch(JAPANESE_CHAR_PATTERN, segment):
            continue  # valid Japanese
        elif re.fullmatch(NUMERIC_PATTERN, segment):
            continue  # valid numeric
        elif segment in ALLOWED_ENGLISH_TERMS:
            continue  # valid allowed English term
        else:
            invalid_segments.append(segment)

    if invalid_segments:
        print(f"[WARNING] 不許可の英語／非日本語表現が含まれています: {invalid_segments}")
        return False

    return True


class StripThinkingParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Remove <think>...</think> content
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def get_interview_response(conversation_history, age=None, gender=None, max_retries=3):
    start_time = time.time()
    prompt = ChatPromptTemplate.from_template(interview_template)
    #chain = prompt | llm | StripThinkingParser()
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
