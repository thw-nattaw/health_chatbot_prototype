import streamlit as st
from chatbot import get_interview_response

st.title("🤖 Patient Interview Chatbot")

# --- Initial setup of session state ---
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = ""
if "end_conversation" not in st.session_state:
    st.session_state.end_conversation = False
if "i" not in st.session_state:
    st.session_state.i = 0
if "age" not in st.session_state:
    st.session_state.age = None
if "gender" not in st.session_state:
    st.session_state.gender = None
if "submitted_basic_info" not in st.session_state:
    st.session_state.submitted_basic_info = False

# --- Step 1: Collect basic patient information ---
if not st.session_state.submitted_basic_info:
    st.session_state.age = st.number_input("患者の年齢を入力してください", min_value=0, max_value=120, step=1)
    st.session_state.gender = st.radio("性別を選択してください", ["男性", "女性", "その他"])

    if st.button("チャットを開始する"):
        st.session_state.submitted_basic_info = True
    st.stop()  # Don't run chat section until info is submitted

# --- Step 2: Display chat history ---
for entry in st.session_state.conversation:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

# --- Step 3: Chat input area ---
if not st.session_state.end_conversation:
    user_input = st.chat_input("Describe your symptoms here...")

    if user_input:
        st.session_state.conversation.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.i += 1
        print(st.session_state.i)

        full_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.conversation])
        ai_response = get_interview_response(
            full_history,
            age=st.session_state.age,
            gender=st.session_state.gender
        )

        st.session_state.conversation.append({"role": "assistant", "content": ai_response})
        with st.chat_message("assistant"):
            st.markdown(ai_response)

# --- Step 4: End interview button ---
if st.button("End Interview"):
    st.session_state.end_conversation = True
    full_convo = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.conversation])
    st.success("Interview ended. Go to the Summary page.")
