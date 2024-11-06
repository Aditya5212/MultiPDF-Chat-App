import os
from langchain_community.docstore.in_memory import InMemoryDocstore
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain 
from langchain_groq import ChatGroq
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.embeddings.base import Embeddings

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode(text)
        return embedding.tolist()

def get_pdf_text(pdf_docs):
    # Load the PDF file
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Handle None return from extract_text
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    chunk = text_splitter.split_text(raw_text)
    return chunk

def get_vectorStore(text_chunks):
    # embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # # Generate embeddings for the text chunks using the encode method
    # embeddings = embedding_model.encode(text_chunks, convert_to_tensor=True)
    # # print(f"Embeddings shape: {embeddings.shape}")
    # index = faiss.IndexFlatL2(embeddings.shape[1])
    # vectorStore = FAISS.from_texts(text_chunks,embedding_model)
    # index.add(embeddings)
    # # Create the vector store with the embeddings and the corresponding texts
    # return vectorStore

    embedding_model = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')
    vectorStore = FAISS.from_texts(text_chunks, embedding_model)
    return vectorStore

def get_conversationChain(vectoreStore):
    load_dotenv()
    model = ChatGroq(
        model="llama3-8b-8192",
        api_key=os.getenv('GROQ_API_KEY')
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        max_memory=5  # Adjust as needed
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectoreStore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("Please upload and process PDFs before asking questions.")
        return
    
    try:
        response = st.session_state.conversation({'question': user_question})
    except Exception as e:
        st.error(f"Error during conversation: {e}")
        return
    
    for i, message in enumerate(response['chat_history']):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
   
    st.set_page_config(page_title="Chat With Multiple PDF", page_icon=":books:")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []    
    
    st.write(css, unsafe_allow_html=True)
    st.title("Chat with multiple PDFs :books:")
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload Your PDFs Here and process", accept_multiple_files=True)
        
        if pdf_docs:
            st.success(f"Uploaded {len(pdf_docs)} PDF(s).")
        else:
            st.warning("Please upload at least one PDF before processing.")
        
        if st.button("process"):
            # Check if PDFs are uploaded
            if not pdf_docs:
                st.error("No PDFs uploaded! Please upload PDFs before processing.")
                return  # Exit the processing

            with st.spinner("Processing..."):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("No text was extracted from the PDFs. Please check the PDFs.")
                    return
                
                # Get the Text chunks
                text_chunks = get_text_chunks(raw_text)
                if not text_chunks:
                    st.error("No text chunks were created from the PDF. Please check the PDF content.")
                    return
                
                # Create vector Store
                vector_store = get_vectorStore(text_chunks)
                st.session_state.conversation = get_conversationChain(vector_store)
                st.success("Processing completed!")


if __name__ == '__main__':
    main()