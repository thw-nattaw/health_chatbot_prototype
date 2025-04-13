import streamlit as st
from utils import summarize_conversation

st.title("📄 Conversation Summary (SOAP)")

if "conversation" not in st.session_state or not st.session_state.conversation:
    st.warning("No conversation available. Please start from the Chatbot page.")
else:
    full_convo = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.conversation])
    summary = summarize_conversation(full_convo)

    st.text_area("SOAP Summary", summary, height=400)

    if st.download_button("💾 Download Summary", summary, file_name="conversation_summary.txt"):
        st.success("Summary downloaded.")

    if st.button("🔁 Restart"):
        for key in ["conversation", "diagnosis", "end_conversation"]:
            st.session_state.pop(key, None)
        st.experimental_rerun()
