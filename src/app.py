import os
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- Page Config ---
st.set_page_config(page_title="DocuChat AI", page_icon="📚")
st.title("📚 Chat with your PDF")

# --- Logic to Switch between Local (Ollama) and Cloud (Hugging Face) ---
# Check if we are running on Hugging Face Spaces by looking for the token
HF_TOKEN = os.getenv("HF_TOKEN")

def get_llm():
    if HF_TOKEN:
        # We are on the Cloud (Hugging Face Space)
        # Uses free Serverless Inference API (Mistral-7B is good and free)
        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        return HuggingFaceEndpoint(
            repo_id=repo_id, 
            task="text-generation", 
            max_new_tokens=512,
            temperature=0.3,
            huggingfacehub_api_token=HF_TOKEN
        )
    else:
        # We are Local
        # Uses your local Ollama instance
        return ChatOllama(model="llama3.1", temperature=0.3)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file and not st.session_state.vector_db:
        with st.spinner("Processing PDF..."):
            # Save file temporarily to process it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # Load and Chunk
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            
            # Embeddings (Use HuggingFace Embeddings for both)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Create DB
            st.session_state.vector_db = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings,
                # In Streamlit Cloud, we can't persist to disk reliably, so we use in-memory
            )
            os.remove(tmp_path)
            st.success("✅ PDF Processed!")

# --- Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the PDF..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if st.session_state.vector_db:
            # 1. Retrieve Context
            retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
            relevant_docs = retriever.invoke(prompt)
            context_text = "\n\n".join([d.page_content for d in relevant_docs])
            
            # 2. Prepare Prompt
            template = """You are a helpful assistant. Answer based on the context provided.
            
            Context: {context}
            
            Question: {question}
            """
            prompt_template = ChatPromptTemplate.from_template(template)
            chain = prompt_template | get_llm()
            
            # 3. Stream Response
            response = chain.invoke({"context": context_text, "question": prompt})
            
            # Handle response format difference (Ollama returns obj, HF returns string sometimes)
            final_response = response.content if hasattr(response, 'content') else response
            st.markdown(final_response)
            
            st.session_state.messages.append({"role": "assistant", "content": final_response})
        else:
            st.warning("⚠️ Please upload a PDF first.")