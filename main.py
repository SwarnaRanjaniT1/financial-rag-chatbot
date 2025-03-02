import os
import streamlit as st
import tempfile
from pathlib import Path
import torch
import socket
from rag_components import FinancialRAG
from guardrails import FinancialGuardrail
from utils import get_confidence_score

# Set page configuration
st.set_page_config(
    page_title="Financial RAG Chatbot",
    page_icon="💼",
    layout="wide"
)

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Running on device: {device}")

# Display environment information
is_streamlit_cloud = os.environ.get('IS_STREAMLIT_CLOUD') == 'true'
if is_streamlit_cloud:
    st.sidebar.success("Running on Streamlit Cloud")
else:
    # In Replit or local environment
    pass

# Initialize session state variables
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# App title and description
st.title("📊 Financial Statement RAG Chatbot")
st.markdown("""
This application uses Retrieval-Augmented Generation (RAG) to answer questions about financial statements.
Upload financial documents (PDFs) to get started.
""")

# Sidebar for document upload and system configuration
st.sidebar.title("Document Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload financial statements (PDF)", 
    type="pdf", 
    accept_multiple_files=True
)

# Model selection
st.sidebar.title("Model Configuration")
embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    options=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L3-v2"],
    index=0
)

llm_model = st.sidebar.selectbox(
    "Language Model",
    options=["google/flan-t5-small", "facebook/opt-125m", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
    index=0
)

# Advanced options
st.sidebar.title("Advanced Options")
chunk_size = st.sidebar.slider("Chunk Size", min_value=128, max_value=1024, value=512, step=64)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=256, value=50, step=8)
top_k = st.sidebar.slider("Number of Retrieved Documents", min_value=1, max_value=10, value=4)

# Initialize or update RAG system
if uploaded_files and st.sidebar.button("Process Documents"):
    with st.spinner("Processing documents... This may take a while."):
        temp_dir = tempfile.mkdtemp()
        pdf_paths = []
        
        # Save uploaded files to temporary directory
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_paths.append(file_path)
        
        # Initialize the RAG system
        st.session_state.rag_system = FinancialRAG(
            embedding_model=embedding_model,
            llm_model=llm_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            device=device
        )
        
        # Process the documents
        st.session_state.rag_system.ingest_documents(pdf_paths)
        st.session_state.documents_processed = True
        st.success(f"Processed {len(pdf_paths)} documents successfully!")

# Initialize guardrail
guardrail = FinancialGuardrail()

# Chat interface
st.header("Ask about the financial statements")

if not st.session_state.documents_processed:
    st.info("Please upload and process financial documents to start chatting.")
else:
    # Display chat history
    for i, (question, answer, confidence) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
            st.caption(f"Confidence: {confidence:.2f}")
    
    # Get user question
    user_question = st.chat_input("Ask a question about the financial statements")
    
    if user_question:
        # Add user question to chat
        with st.chat_message("user"):
            st.write(user_question)
        
        # Check if question passes guardrail
        is_valid, reason = guardrail.validate_input(user_question)
        
        with st.chat_message("assistant"):
            if is_valid:
                with st.spinner("Generating answer..."):
                    # Get answer from RAG system
                    answer, sources = st.session_state.rag_system.answer_question(user_question, top_k=top_k)
                    
                    # Calculate confidence score
                    confidence = get_confidence_score(user_question, answer, sources)
                    
                    # Display answer and confidence
                    st.write(answer)
                    st.caption(f"Confidence: {confidence:.2f}")
                    
                    # Display sources if confidence is low
                    if confidence < 0.7:
                        with st.expander("Sources"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**Source {i+1}:**")
                                st.markdown(source.page_content)
                                st.markdown("---")
                    
                    # Add to chat history
                    st.session_state.chat_history.append((user_question, answer, confidence))
            else:
                st.write(f"❌ {reason} Please ask a question related to the financial statements.")

# Add information about the system
st.sidebar.title("About")
st.sidebar.info("""
This RAG system implements Hybrid Search (combining Sparse BM25 and Dense Vector retrieval) 
to improve answer quality. It leverages both keyword-based and semantic matching
to find the most relevant context for answering financial questions.

It uses open-source embedding models and language models to provide
accurate answers to questions about financial statements.
""")

# Add hostname and connection information
if not is_streamlit_cloud:
    try:
        hostname = socket.gethostname()
        host_ip = socket.gethostbyname(hostname)
        
        st.sidebar.title("Connection Information")
        st.sidebar.info(f"""
        **Hostname:** {hostname}
        **IP Address:** {host_ip}
        **Port:** {os.environ.get('PORT', '5000')}
        
        To access this app from outside Replit:
        1. Deploy the app to a cloud service like Heroku, AWS, or Azure
        2. Or use Streamlit Cloud (streamlit.io/cloud)
        3. Or self-host on your own server using `streamlit run main.py --server.address=0.0.0.0 --server.port=80`
        """)
    except Exception as e:
        pass

# App footer
st.markdown("---")
st.markdown("Financial RAG Chatbot - Built with Streamlit and open-source models")
