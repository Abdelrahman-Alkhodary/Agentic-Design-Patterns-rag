import yaml
from src.database.postgres import hybrid_search
import streamlit as st
from src.llm.openai import OpenaiLLM


# Initialize client
llm = OpenaiLLM()

st.set_page_config(page_title="Q&A App", layout="wide")

# --- Session State ---
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

st.title("Q&A Assistant For the Book Agentic Design Patterns")

# --- Display previous Q&A at top ---
if st.session_state.last_question:
    st.subheader("Question:")
    st.info(st.session_state.last_question)

    st.subheader("Bot Answer:")
    st.success(st.session_state.last_answer)

# Spacer to push input to bottom
st.markdown("<div style='height: 45vh;'></div>", unsafe_allow_html=True)

# --- Input at bottom ---
with st.container():
    st.markdown("### Ask your question")
    question = st.text_area(" ", placeholder="Type your question here...", key="question_box")
    
    if st.button("Submit"):
        if question.strip():
            with st.spinner("Searching knowledge base..."):
                # MOVED INSIDE: Only run search if question exists
                results = hybrid_search(query_text=question, embedding_fn=llm.get_embedding)
                
                retrieved_texts = ""
                for i, res in enumerate(results):
                    content = res['data'][1]
                    retrieved_texts += content + "\n\n"
                
                print(f"retrieved text: {retrieved_texts}")

            
            with st.spinner("Generating answer..."):
                # Using standard ChatCompletions (Adjust if using a custom wrapper)
                answer = llm.generate_answer(question=question, retrieved_texts=retrieved_texts)

                # Store in session state
                st.session_state.last_question = question
                st.session_state.last_answer = answer

                # Force reload to show Q&A at top
                st.rerun()
        else:
            st.warning("Please enter a question.")