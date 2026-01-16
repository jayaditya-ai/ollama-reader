"""
PDF RAG System - Modern Web UI
Professional chat interface for PDF question answering using Ollama
"""

import os
import sys
import requests
from pathlib import Path
from typing import List
import streamlit as st
import time

try:
    from PyPDF2 import PdfReader
    import chromadb
    from chromadb.config import Settings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    st.error("Installing required packages... Please refresh the page after installation completes.")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "PyPDF2", "chromadb", "langchain-text-splitters", "requests", "streamlit"])
    st.rerun()


# Load custom CSS from external file
def load_custom_css():
    css_path = Path(__file__).parent / "style.css"

    with open(css_path, "r") as f:
        css_content = f.read()

    st.markdown(f"""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* Custom header */
    .main-header {{
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
    }}

    .main-header h1 {{
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}

    .main-header p {{
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
    }}

    {css_content}
    </style>
    """, unsafe_allow_html=True)


class PDFRAGSystem:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model = "llama3.2"
        self.embed_model = "nomic-embed-text"

        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory="./pdf_db"
        ))

        try:
            self.collection = self.client.get_collection("pdfs")
        except:
            self.collection = self.client.create_collection("pdfs")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def check_ollama(self):
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            return True, "Ollama is running"
        except Exception as e:
            return False, f"Ollama is not running. Please run: docker-compose up -d"

    def check_model(self, model_name):
        """Check if a model exists."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            models = response.json().get("models", [])
            return any(model_name in m["name"] for m in models)
        except:
            return False

    def pull_model(self, model_name):
        """Download a model from Ollama."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600
            )
            return True
        except Exception as e:
            return False

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings from Ollama."""
        response = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={"model": self.embed_model, "prompt": text}
        )
        return response.json()["embedding"]

    def process_pdf(self, pdf_file):
        """Process uploaded PDF file."""
        # Read PDF
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        # Split into chunks
        chunks = self.splitter.split_text(text)

        # Store in database
        pdf_name = pdf_file.name.replace(".pdf", "")
        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk)
            self.collection.add(
                embeddings=[embedding],
                documents=[chunk],
                ids=[f"{pdf_name}_{i}"],
                metadatas=[{"source": pdf_name, "chunk": i}]
            )

        return len(chunks)

    def ask_question(self, question: str, n_results: int = 3) -> tuple:
        """Ask a question and get answer with context."""
        # Get relevant context
        query_embedding = self.get_embedding(question)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        if not results["documents"][0]:
            return "No information found. Please upload a PDF first.", []

        context_docs = results["documents"][0]
        context = "\n\n".join(context_docs)

        # Generate answer
        prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )

        return response.json()["response"], context_docs


# Main UI
def main():
    st.set_page_config(
        page_title="PDF Question Answering",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load custom CSS
    load_custom_css()

    # Initialize session state
    if 'rag' not in st.session_state:
        st.session_state.rag = PDFRAGSystem()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'pdf_loaded' not in st.session_state:
        st.session_state.pdf_loaded = False

    if 'pdf_name' not in st.session_state:
        st.session_state.pdf_name = None

    # Custom header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 PDF Question Answering</h1>
        <p>Powered by Ollama • 100% Local • Zero API Costs</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ System Status")

        # Check Ollama
        is_running, message = st.session_state.rag.check_ollama()
        if is_running:
            st.markdown('<div class="status-badge status-success">✓ Ollama Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-error">✗ Ollama Offline</div>', unsafe_allow_html=True)
            st.error(message)
            st.code("docker-compose up -d", language="bash")
            st.stop()

        st.markdown("---")

        # Check models
        st.markdown("### 🤖 AI Models")

        chat_model_exists = st.session_state.rag.check_model("llama3.2")
        embed_model_exists = st.session_state.rag.check_model("nomic-embed-text")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Chat Model**")
        with col2:
            if chat_model_exists:
                st.markdown("✓")
            else:
                st.markdown("✗")

        if not chat_model_exists:
            if st.button("📥 Download llama3.2", use_container_width=True):
                with st.spinner("Downloading (~2GB)..."):
                    st.session_state.rag.pull_model("llama3.2")
                    st.rerun()

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Embedding Model**")
        with col2:
            if embed_model_exists:
                st.markdown("✓")
            else:
                st.markdown("✗")

        if not embed_model_exists:
            if st.button("📥 Download nomic-embed", use_container_width=True):
                with st.spinner("Downloading (~700MB)..."):
                    st.session_state.rag.pull_model("nomic-embed-text")
                    st.rerun()

        if not (chat_model_exists and embed_model_exists):
            st.stop()

        st.markdown("---")

        # PDF Upload
        st.markdown("### 📄 Upload Document")
        uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

        if uploaded_file is not None:
            if st.button("🚀 Process PDF", use_container_width=True, type="primary"):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        num_chunks = st.session_state.rag.process_pdf(uploaded_file)
                        st.session_state.pdf_loaded = True
                        st.session_state.pdf_name = uploaded_file.name
                        st.session_state.messages = []
                        st.success(f"✓ Processed {num_chunks} chunks!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state.pdf_loaded:
            st.markdown('<div class="status-badge status-success">📗 PDF Ready</div>', unsafe_allow_html=True)
            st.caption(f"📄 {st.session_state.pdf_name}")

        st.markdown("---")

        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Footer
        st.markdown("---")
        st.caption("Built with Streamlit & Ollama")
        st.caption("100% Local • Private • Secure")

    # Main content area
    if not st.session_state.pdf_loaded:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 👋 Welcome!")
            st.markdown("""
            Get started in 3 simple steps:

            1. **Upload** - Choose a PDF from the sidebar
            2. **Process** - Click the Process PDF button
            3. **Ask** - Type your questions below

            Your data never leaves your machine!
            """)

            with st.expander("💡 Example Questions"):
                st.markdown("""
                - What is this document about?
                - Summarize the main points
                - What does it say about [topic]?
                - List the key findings
                - Explain the methodology
                """)
    else:
        # Chat interface
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])

                # Show context for assistant messages
                if message["role"] == "assistant" and "context" in message:
                    with st.expander("📄 View Source Context"):
                        for i, ctx in enumerate(message["context"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.caption(ctx[:300] + "..." if len(ctx) > 300 else ctx)
                            if i < len(message["context"]):
                                st.markdown("---")

        # Chat input
        if prompt := st.chat_input("Ask a question about your document..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    try:
                        answer, context = st.session_state.rag.ask_question(prompt)
                        st.markdown(answer)

                        # Store in messages
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "context": context
                        })

                        # Show context
                        with st.expander("📄 View Source Context"):
                            for i, ctx in enumerate(context, 1):
                                st.markdown(f"**Source {i}:**")
                                st.caption(ctx[:300] + "..." if len(ctx) > 300 else ctx)
                                if i < len(context):
                                    st.markdown("---")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.session_state.messages.pop()  # Remove failed user message


if __name__ == "__main__":
    main()
