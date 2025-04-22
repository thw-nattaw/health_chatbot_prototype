import streamlit as st
from chatbot import get_interview_response

st.title("🤖 Patient Interview Chatbot")

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = ""
if "end_conversation" not in st.session_state:
    st.session_state.end_conversation = False
if "i" not in st.session_state:
    st.session_state.i = 0

# Display previous messages
for entry in st.session_state.conversation:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

# Bottom input
if not st.session_state.end_conversation:
    user_input = st.chat_input("Describe your symptoms here...")

    if user_input:
        st.session_state.conversation.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.i = st.session_state.i + 1
        print(st.session_state.i)
        
        # Full context for LLM
        full_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.conversation])
        ai_response = get_interview_response(user_input, full_history)

        st.session_state.conversation.append({"role": "assistant", "content": ai_response})
        with st.chat_message("assistant"):
            st.markdown(ai_response)

# End button
if st.button("End Interview"):
    st.session_state.end_conversation = True
    full_convo = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.conversation])
    st.success("Interview ended. Go to the Summary page.")
