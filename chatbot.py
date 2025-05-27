from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
import time
import re

llm = OllamaLLM(model="qwen3:32b")

detail = "Y"
dx = "Y"
think = True

def load_prompt(filename):
    with open(f"prompts/{filename}.txt", "r", encoding="utf-8") as f:
        return f.read()

def get_prompt(detail, dx, think):
    # Determine which prompt to use
    if detail == "N" and dx == "N":
        prompt_name = "prompt1"
    elif detail == "Y" and dx == "N":
        prompt_name = "prompt2"
    elif detail == "N" and dx == "Y":
        prompt_name = "prompt3"
    elif detail == "Y" and dx == "Y":
        prompt_name = "prompt4"
    else:
        raise ValueError("Invalid combination of A and B")

    prompt = load_prompt(prompt_name)

    if not think:
        prompt += "\n/no_think"

    return prompt

interview_prompt = get_prompt(detail, dx, think)

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


class StripThinkingParserWithLogging(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Extract <think> part (if exists)
        match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if match:
            thought = match.group(1).strip()
            print("MODEL THOUGHT:\n", thought)
        else:
            print("MODEL THOUGHT: (None found)")
        
        # Return only the visible output
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def get_interview_response(conversation_history, age=None, gender=None, max_retries=3):
    start_time = time.time()
    prompt = ChatPromptTemplate.from_template(interview_prompt)
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
