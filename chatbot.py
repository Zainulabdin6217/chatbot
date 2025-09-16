from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import tempfile
import os


# ----------- Chunking Function -----------
def chunking(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50
    )
    return text_splitter.split_documents(data)


# ----------- LLM Response Function -----------
def get_llm_response(query, content):
    llm = ChatOpenAI(
        model="gpt-4o-mini",   # or "gpt-4o" for stronger reasoning
        temperature=0.3
    )
    template = """
    You have to answer the user query based on the provided context.
    If the user asks something that is out of context of the content,
    then guide the user on what he can ask and what you can tell him.

    User query: {query}
    Context: {content}
    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": query, "content": content})
    return response


# ----------- Vector Store Function -----------
def store_in_vector(chunks):
    if not chunks:
        st.error("No content to process. Please check your files.")
        return None
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


# ----------- Retriever Function -----------
def retrieving(vector_store, query):
    if vector_store is None:
        return []
    return vector_store.similarity_search(query, k=3)


# ----------- File Processing Function -----------
def process_file(uploaded_files):
    with st.spinner("Processing..."):
        all_documents = []
        for uploaded_file in uploaded_files:
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())

            ext = Path(path).suffix.lower()
            if ext == ".csv":
                loader = CSVLoader(path)
            elif ext == ".pdf":
                loader = PyPDFLoader(path)
            elif ext == ".txt":
                loader = TextLoader(path, encoding="utf-8")
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
                continue

            try:
                data = loader.load()
                if not data:
                    st.warning(f"No data found in file: {uploaded_file.name}")
                all_documents.extend(data)
            except Exception as e:
                st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        
        if not all_documents:
            st.error("No valid data found in any of the uploaded files.")
            return None
        
        chunks = chunking(all_documents)
        return store_in_vector(chunks)
    
    


# ----------- Main Streamlit App -----------
def main():
    load_dotenv()
    st.set_page_config(page_title="Document Talks")
    st.header("📄 Chit-Chat with your File")

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:
        st.subheader("Upload your File(s)")
        uploaded_files = st.file_uploader(
            "Upload CSV, PDF, or TXT files...",
            type=["csv", "pdf", "txt"],
            accept_multiple_files=True
        )
        if uploaded_files and st.session_state.vector_store is None:
            st.session_state.vector_store = process_file(uploaded_files)
            if st.session_state.vector_store:
                st.success("File(s) Uploaded and Processed Successfully ✅")
            else:
                st.error("Failed to process files. Please check the errors above.")

    input_text = st.text_input("Ask anything about your file(s): ")
    if input_text and st.session_state.vector_store:
        with st.spinner("Thinking..."):
            similar_chunk = retrieving(st.session_state.vector_store, input_text)
            if similar_chunk:
                response = get_llm_response(input_text, similar_chunk)
                st.write(response)
            else:
                st.warning("No relevant information found. Try a different question or upload more files.")


if __name__ == "__main__":
    main()
