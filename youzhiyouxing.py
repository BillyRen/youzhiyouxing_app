import os
import openai
import pandas as pd
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import PyPDF2

# Define the app title and icon
st.set_page_config(page_title="有知有行投资问答机器人", page_icon=':books:')

### Functions ###

@st.cache_data()
def load_pdf(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@st.cache_resource()
def retriever_function(loaded_text, chunk_size, overlap_size):    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        length_function=len
    )
    doc = text_splitter.create_documents([loaded_text])
    if not doc:
        st.error("Failed to split the pdf")
        st.stop()

    search_index = FAISS.from_documents(doc, OpenAIEmbeddings())
    return search_index

### Main Function ###
def main():
    # Hide Streamlit branding
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .css-1y0tads {display: none !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # App header
    st.markdown(
        """
        <div style="display: flex; align-items: center; margin-left: 0;">
            <h1 style="display: inline-block;">有知有行投资问答机器人</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize question in session state
    if 'question' not in st.session_state:
        st.session_state.question = ""

    def submit():
        st.session_state.question = st.session_state.widget
        st.session_state.widget = ""

    st.text_input("Enter your question:", key="widget", on_change=submit)

    # Initialize history
    if 'history' not in st.session_state:
        st.session_state['history'] = pd.DataFrame(columns=['Question', 'Answer', 'Pdf'])

    # Sidebar
    st.sidebar.title("Parameters")
    api_key = os.getenv('OPENAI_API_KEY_SC')
    chunk_size = st.sidebar.number_input('Chunk size (characters)', value=20000)
    overlap_size = st.sidebar.number_input('Overlap size (characters)', value=2000)

    # Read PDF file
    pdf_path = 'data/youzhiyouxing_v2.0.pdf'
    if not os.path.exists(pdf_path):
        st.error("Specified PDF file not found.")
        st.stop()

    with open(pdf_path, 'rb') as file:
        loaded_text = load_pdf(file)
      
    # Handle API key
    if api_key is None:
        st.warning("Please add a key")
        return
    else:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Create embeddings
    search_index = retriever_function(loaded_text=loaded_text, chunk_size=chunk_size, overlap_size=overlap_size)

    # Question-answering logic
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-4-1106-preview"), retriever=search_index.as_retriever())

    if st.session_state.question:
        user_question = st.session_state.question
        answer = qa.run(user_question)
        st.write(f"{user_question}: {answer}")
        st.session_state.question = ""
        st.session_state['history'].loc[len(st.session_state['history'])] = [user_question, answer, pdf_path]

    # Print history except current question
    if len(st.session_state['history']) > 1:
        print_history = st.session_state['history'][:-1].tail(3)
        st.markdown("<br><br><br><p style='font-size: 30px; text-align:center'><b>HISTORY</b></p>", unsafe_allow_html=True,)
        for index, row in reversed(print_history.iterrows()):
            st.markdown(f"<br><hr><br><p><b>{row['Question']}</b>:<br></p><p>{row['Answer']} </p><p style='font-size: 14px;'>{row['Pdf']}</p>", unsafe_allow_html=True,)

if __name__ == "__main__":
    main()