import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template
import logging
import faiss

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            logging.error(f"Error processing PDF file: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    logging.info("Starting vector store creation")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    logging.info("Embeddings created")
    
    # Create the FAISS vector store
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    logging.info("FAISS vector store created")

    # Define the directory where the FAISS index will be saved
    faiss_index_dir = os.path.join(os.path.dirname(__file__), "faiss_index")
    os.makedirs(faiss_index_dir, exist_ok=True)

    # Save the entire FAISS vector store, including the docstore and index_to_docstore_id
    vector_store.save_local(faiss_index_dir)
    logging.info("FAISS vector store saved successfully.")

def get_conversation_chain():
    prompt_template = """
        Answer the question clear and precise. If not provided the context return the result as
        "Sorry I dont know the answer", don't provide the wrong answer.
        Context:\n {context}?\n
        Question:\n{question}\n
        Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain

def user_input(user_question):
    logging.info("Processing user input")
    
    # Reload the FAISS vector store from the saved directory
    faiss_index_dir = os.path.join(os.path.dirname(__file__), "faiss_index")
    
    if not os.path.exists(faiss_index_dir):
        st.warning("Please upload and process PDF files before asking questions.")
        return

    try:
        # Load the entire FAISS vector store, enabling dangerous deserialization since we trust the source
        new_db = FAISS.load_local(faiss_index_dir, GoogleGenerativeAIEmbeddings(model='models/embedding-001'), allow_dangerous_deserialization=True)
        logging.info("FAISS vector store loaded successfully")
        
        # Perform similarity search and generate response
        docs = new_db.similarity_search(user_question)
        chain = get_conversation_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write(user_template.replace("{{MSG}}", response["output_text"]), unsafe_allow_html=True)
    except Exception as e:
        logging.error(f"Error processing user input: {e}")
        st.write(bot_template.replace("{{MSG}}", f"Sorry, there was an error processing your request: {str(e)}. Please try again later."), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs with Gemini Pro :books:")
    
    with st.sidebar:
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on Process",
            accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversation_chain()
                    st.success("PDFs processed successfully. You can now ask questions.")
                except Exception as e:
                    logging.error(f"Error processing PDF files: {e}")
                    st.error("There was an error processing the PDF files. Please try again later.")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "faiss_index", "index.faiss")):
            st.warning("Please upload and process PDF files before asking questions.")
        else:
            user_input(user_question)

if __name__ == "__main__":
    main()