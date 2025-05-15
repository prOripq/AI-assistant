import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
import tempfile
import time
from functools import wraps
import hashlib
from typing import List, Dict, Any

# Constants
CONSTITUTION_URL = "https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PERSIST_DIR = "./chroma_db"
MODEL_NAME = "mistral"  # Consider using "llama2" or "mistral" for better accuracy

# Setup
load_dotenv()
set_llm_cache(InMemoryCache())

# Embeddings & LLM with temperature control for more deterministic answers
embeddings = OllamaEmbeddings(model=MODEL_NAME)
llm = Ollama(
    model=MODEL_NAME,
    temperature=0.3,  # Lower temperature for more factual answers
    top_k=10,        # Controls diversity
    top_p=0.9        # Nucleus sampling
)

class DocumentProcessor:
    @staticmethod
    @st.cache_data(show_spinner="Loading and processing the Constitution...")
    def load_constitution() -> List[Dict[str, Any]]:
        """Load and split the constitution document with error handling"""
        try:
            loader = WebBaseLoader(CONSTITUTION_URL)
            raw_docs = loader.load()
            return DocumentProcessor.split_documents(raw_docs)
        except Exception as e:
            st.error(f"Failed to load constitution: {str(e)}")
            return []

    @staticmethod
    @st.cache_data
    def split_documents(_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents with optimal chunking for legal text"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # Better for legal docs
        )
        return text_splitter.split_documents(_docs)

    @staticmethod
    def process_uploaded_files(uploaded_files: List[Any]) -> List[Dict[str, Any]]:
        """Process uploaded files with robust error handling"""
        documents = []
        for uploaded_file in uploaded_files:
            try:
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                if file_ext == '.pdf':
                    loader = PyPDFLoader(temp_file_path)
                else:
                    loader = TextLoader(temp_file_path)
                
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
            except Exception as e:
                st.warning(f"Could not process {uploaded_file.name}: {str(e)}")
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        return documents

class VectorStoreManager:
    @staticmethod
    def get_vectorstore(documents: List[Dict[str, Any]], persist_dir: str = PERSIST_DIR) -> Chroma:
        """Get or create vector store with proper indexing"""
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            return Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
        else:
            return Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_metadata={"hnsw:space": "cosine"}  # Better for semantic similarity
            )

class QASystem:
    @staticmethod
    def create_retrieval_chain(retriever: Any) -> Any:
        """Create a more sophisticated QA chain for legal documents"""
        template = """You are an AI legal assistant specializing in the Constitution of Kazakhstan and related legal documents. 
        Your answers must be accurate, precise, and based only on the provided context.

        Context:
        {context}

        Question: {question}

        Guidelines for your response:
        1. Answer clearly and concisely
        2. Always cite specific articles/sections when possible (e.g., "According to Article 15.2...")
        3. If the question is unclear, ask for clarification
        4. If you don't know, say "I don't know" - don't make up answers
        5. For comparative questions, highlight differences/similarities
        6. For procedural questions, outline steps clearly
        7. Format your response with clear paragraphs and bullet points when appropriate"""

        prompt = ChatPromptTemplate.from_template(template)
        
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    @staticmethod
    def rate_limit(max_calls: int = 3, period: int = 60):
        """Enhanced rate limiter with request hashing"""
        def decorator(f):
            calls = {}
            @wraps(f)
            def wrapper(*args, **kwargs):
                # Create a unique hash for each question to prevent duplicate processing
                question_hash = hashlib.md5(kwargs.get('question', '').encode()).hexdigest()
                now = time.time()
                
                # Clear old calls
                calls[question_hash] = [call for call in calls.get(question_hash, []) if call > now - period]
                
                if len(calls[question_hash]) >= max_calls:
                    wait_time = period - (now - calls[question_hash][0])
                    st.warning(f"Rate limit reached for this question. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    calls[question_hash] = []
                
                calls[question_hash].append(now)
                return f(*args, **kwargs)
            return wrapper
        return decorator

    @rate_limit(max_calls=3, period=60)
    def get_answer(chain: Any, question: str) -> str:
        """Get answer with error handling and fallback"""
        try:
            return chain.invoke(question)
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return "I encountered an error while processing your question. Please try again."

def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.constitution_loaded = False
        st.session_state.files_processed = False
        st.session_state.vectorstore = None

def display_chat_history():
    """Display chat messages with proper formatting"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    st.markdown(item)
            else:
                st.markdown(msg["content"])

def main():
    st.set_page_config(
        page_title="🇰🇿 AI Assistant for Kazakhstan Constitution",
        page_icon="🇰🇿",
        layout="wide"
    )
    
    st.title("🇰🇿 AI Assistant for Kazakhstan Constitution")
    st.caption("Ask questions about the Constitution of Kazakhstan and related legal documents")
    
    initialize_session_state()
    
    # Load Constitution once
    if not st.session_state.constitution_loaded:
        with st.spinner("Loading and indexing the Constitution of Kazakhstan..."):
            constitution_docs = DocumentProcessor.load_constitution()
            if constitution_docs:
                st.session_state.vectorstore = VectorStoreManager.get_vectorstore(constitution_docs)
                st.session_state.constitution_loaded = True
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Hello! I'm your AI legal assistant for the Constitution of Kazakhstan. I can help you with:"
                    "\n- Explaining constitutional articles"
                    "\n- Analyzing legal concepts"
                    "\n- Comparing provisions"
                    "\n- Answering questions about rights and procedures"
                })

    # File upload section
    with st.expander("📁 Upload Additional Legal Documents", expanded=False):
        uploaded_files = st.file_uploader(
            "Select PDF or text files to enhance my knowledge",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files and not st.session_state.files_processed:
            with st.spinner("Processing and indexing uploaded documents..."):
                uploaded_docs = DocumentProcessor.process_uploaded_files(uploaded_files)
                if uploaded_docs:
                    split_docs = DocumentProcessor.split_documents(uploaded_docs)
                    st.session_state.vectorstore = VectorStoreManager.get_vectorstore(split_docs)
                    st.session_state.files_processed = True
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Successfully processed {len(uploaded_files)} document(s). "
                                   "You can now ask questions about these documents along with the Constitution."
                    })

    display_chat_history()

    # Chat input with example questions
    if prompt := st.chat_input("Ask a question about the Constitution..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            if not st.session_state.vectorstore:
                st.error("Document processing is not complete yet. Please wait...")
                return
                
            with st.spinner("Researching your question..."):
                try:
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr",  # Maximal Marginal Relevance for better diversity
                        search_kwargs={"k": 5}  # Get 5 most relevant chunks
                    )
                    chain = QASystem.create_retrieval_chain(retriever)
                    response = QASystem.get_answer(chain, prompt)
                    
                    # Post-process the response for better formatting
                    formatted_response = response.replace("Answer:", "").strip()
                    formatted_response = formatted_response.replace("\n", "  \n")  # Better markdown formatting
                    
                    st.markdown(formatted_response)
                    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
