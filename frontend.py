import streamlit as st
import requests

st.title("Local RAG Chatbot with FastAPI & Streamlit")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        try:
            response = requests.post("http://localhost:8000/query/", json={"query": query})
            if response.status_code == 200:
                answer = response.json().get("answer")
                st.write("Answer:")
                st.write(answer)
            else:
                st.error("Failed to get response from backend")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
