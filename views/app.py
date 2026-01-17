import streamlit as st
import tempfile
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.controllers.main_controller import MainController

st.set_page_config(page_title="Neura Dynamics Agent", page_icon="🤖", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    .stMarkdown h1 {
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_controller():
    return MainController()

try:
    controller = get_controller()
except Exception as e:
    st.error(f"Failed to initialize controller: {e}")
    st.stop()

st.title("NeuraDynamics AI Agent")
st.markdown("ask me about the **Weather** or your **Documents**.")

# Sidebar for PDF Upload
with st.sidebar:
    st.header("📂 Knowledge Base")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        if st.button("Process PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            with st.spinner("Processing PDF..."):
                try:
                    msg = controller.upload_pdf(tmp_path)
                    st.success(msg)
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                finally:
                    # Cleanup
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask something..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response, state = controller.handle_query(prompt)
                st.markdown(response)
                
                # Debug Info
                with st.expander("🛠️ Debug Info"):
                    st.write(f"**Intent:** {state.get('intent', 'Unknown')}")
                    if state.get('weather_data'):
                        st.json(state['weather_data'])
                    if state.get('rag_context'):
                        st.write("Retrieved Chunks:")
                        for doc in state['rag_context']:
                            st.caption(doc.page_content[:200] + "...")
            except Exception as e:
                response = f"An error occurred: {e}"
                st.error(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})
