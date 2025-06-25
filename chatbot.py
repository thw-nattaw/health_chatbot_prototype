from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import BaseOutputParser
import time
import re

llm = OllamaLLM(model="qwen3:32b")

detail = True
think = True

system_prompt_1 = """You are an AI-powered virtual medical assistant conducting a patient interview in Japanese.
Instruction:
- Output only **one** question per response in Japanese. 
- End the conversation when appropriate."""

system_prompt_2 = """You are an AI-powered virtual medical assistant conducting a patient interview in Japanese.

Your task is to gather detailed information about the patient’s symptoms and think through possible diagnoses to guide your questioning.

Instruction:
- Generate only the question content.
- Do **not** include any prefixes like '質問：', '医師：'.
- Output only **one** question per response in Japanese. 
- Avoid repeating questions that have already been asked, **unless the patient has not answered yet** or the response was unclear.
- Ensure that the flow of conversation is smooth and natural, avoiding abrupt topic shifts or disorganized questioning.
- Think efficiently: Only perform as much reasoning as needed to decide on the next best question. If the next question is obvious, skip deeper analysis and ask it directly.

Think step by step — but be flexible and efficient:
1. First, determine why the patient came today.
   - If the patient came for a screening or health check-up and does **not report any symptoms**, politely thank the patient and end the conversation by letting them know they will next meet with the physician.
   - If the patient has any symptoms, proceed with a detailed interview.
2. Identify and clarify the main symptom if present.
3. Before moving on to associated symptoms, gather a full set of details for each reported symptom:
   - Onset
   - Location
   - Quality (e.g., sharp, dull, throbbing)
   - Severity
   - Duration and course
   - Triggers and relieving factors
   - Recurrence history
   - Past treatments and their effects
4. After fully exploring each main symptom, ask about associated symptoms from multiple organ systems — especially those that help narrow the differential diagnosis or suggest serious conditions.
5. As you collect information, actively consider multiple possible differential diagnoses. Do not stop at just one explanation. Stay broad and open in your reasoning.
6. Ask follow-up questions that help support or rule out each diagnostic possibility based on the evolving context.
7. Explore external or environmental causes (e.g., trauma, infection exposure, allergens) as needed.
8. Ask about relevant risk factors (e.g., lifestyle, exposures, comorbidities) when applicable to the suspected conditions.
9. Ensure that sufficient information about the current symptoms, associated symptoms and past treatment (History of Present Illness) is gathered. Then proceed to ask about the following in this fixed order:
   - Past medical history, treatment, and compliance
   - Family history
   - Smoking and alcohol use
   - Then conclude the conversation by politely informing the patient that all necessary information has been collected, thank them for their cooperation, and let them know they will next see the doctor. Ask about expectation and other questions they want to ask the physician.
"""

system_prompt = system_prompt_2 if detail else system_prompt_1
system_prompt += "/no_think" if not think else ""


human_prompt = """Patient Information:
- 年齢: {age}歳
- 性別: {gender}

Patient Interview Transcript:
{conversation_history}
"""


ALLOWED_ENGLISH_TERMS = {
    "COPD", "NSAIDs", "CT", "MRI", "KG", "BMI"
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


class StripThinkingParserWithLogging(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Extract <think> part (if exists)
        match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if match:
            thought = match.group(1).strip()
            print(thought)
        else:
            print("MODEL THOUGHT: (None found)")
        
        # Return only the visible output
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def get_interview_response(conversation_history, age=None, gender=None, max_retries=3):
    start_time = time.time()
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(human_prompt)
        ])
    chain = prompt | llm | StripThinkingParserWithLogging()

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





# Prompt for generating summary from conversation
summary_template = """
You are a Japanese medical assistant tasked with summarizing a patient interview.

Given the following conversation between the patient and AI assistant, generate a History of Present Illness in the medical records、in Japanese Language.

Conversation:
{conversation_history}
"""

def summarize_conversation(conversation_history: str) -> str:
    """Generate a summary from the patient-AI conversation"""
    prompt = ChatPromptTemplate.from_template(summary_template )
    chain = prompt | llm 
    return chain.invoke({"conversation_history": conversation_history})
