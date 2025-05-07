import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms import Ollama
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain_community.embeddings import OllamaEmbeddings
import time
from functools import wraps
import tempfile

set_llm_cache(InMemoryCache())

# Add rate limiting

def rate_limit(max_calls=3, period=60):
    def decorator(f):
        calls = []
        
        @wraps(f)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls_in_period = [call for call in calls if call > now - period]
            
            if len(calls_in_period) >= max_calls:
                time_to_wait = period - (now - calls_in_period[0])
                st.warning(f"Rate limit reached. Waiting {time_to_wait:.1f} seconds...")
                time.sleep(time_to_wait)
                calls.clear()
            
            calls.append(time.time())
            return f(*args, **kwargs)
        return wrapper
    return decorator

# Decorate your question-answering function
@rate_limit(max_calls=3, period=60)
def get_answer(question):
    # Your existing question-answering logic
    return chain.invoke(question)

# Load environment variables
load_dotenv()

# Initialize components
embeddings = OpenAIEmbeddings()
llm = Ollama(model="mistral")
embeddings = OllamaEmbeddings(model="mistral")

# Load the Constitution of Kazakhstan from the official website
def load_constitution():
    constitution_url = "https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912"
    loader = WebBaseLoader(constitution_url)
    return loader.load()

# Process uploaded files
def process_uploaded_files(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(temp_file_path)
        else:
            loader = TextLoader(temp_file_path)
        
        documents.extend(loader.load())
        os.unlink(temp_file_path)
    return documents

# Split documents into chunks
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(docs)

# Initialize or get the vector store
def get_vector_store(documents):
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

# Create the retrieval chain
def create_retrieval_chain(retriever):
    template = """You are an AI assistant specialized in the Constitution of Kazakhstan and related legal documents.
    Answer the question based only on the following context:
    {context}
    
    Question: {question}
    
    Answer in a clear and concise manner. If you don't know the answer, say you don't know. 
    Always cite the relevant article or section when possible."""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Main Streamlit app
def main():
    st.title("🇰🇿 AI Assistant for Kazakhstan Constitution")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.constitution_loaded = False
    
    # Load the Constitution initially
    if not st.session_state.constitution_loaded:
        with st.spinner("Loading the Constitution of Kazakhstan..."):
            constitution_docs = load_constitution()
            all_docs = split_documents(constitution_docs)
            st.session_state.vectorstore = get_vector_store(all_docs)
            st.session_state.constitution_loaded = True
            st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your AI assistant for the Constitution of Kazakhstan. How can I help you today?"})
    
    # File uploader for additional documents
    uploaded_files = st.file_uploader(
        "Upload additional documents (PDF or TXT)", 
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files and not st.session_state.get("files_processed", False):
        with st.spinner("Processing uploaded files..."):
            uploaded_docs = process_uploaded_files(uploaded_files)
            if uploaded_docs:
                split_uploaded_docs = split_documents(uploaded_docs)
                all_docs = split_uploaded_docs
                st.session_state.vectorstore = get_vector_store(all_docs)
                st.session_state.messages.append({"role": "assistant", "content": f"Processed {len(uploaded_files)} uploaded document(s). You can now ask questions about them."})
                st.session_state.files_processed = True
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the Constitution"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = st.session_state.vectorstore.as_retriever()
                chain = create_retrieval_chain(retriever)
                response = chain.invoke(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
