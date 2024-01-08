import os
#pip3 install 'openai==0.27.0'   
import openai
import pandas
import sys
import numpy as np
import streamlit as st
#langchain. Version used: -> pip3 install 'langchain==0.0.181'
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import time
import logging

#define the app title
st.set_page_config(page_title="有知有行投资问答机器人",page_icon=':books:')

###functions

#caching the data output
@st.cache_data
def load_pdf(file):
#the part below is almost exactly the same as in the Q&A langchain lesson
#we are just adding a basic check to make sure the extension is pdf
#to add other file types simply add if extenson = ".txt" -> that type loader, etc. 
    file_extension = os.path.splitext(file.name)[1]
    if file_extension == ".pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()                
    return text
#####

#cache model
@st.cache_resource
def retriever_function(loaded_text, chunk_size, overlap_size):
    
    #text splitter params
    text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = chunk_size,
                    chunk_overlap  = overlap_size,
                    length_function = len
                    )
    #split the doc
    doc = text_splitter.create_documents([loaded_text])
    #make sure splitting worked. Adding this for other steps would be useful
    #but I am not doing it to not make the code unnecessarily long and to focus on the key concepts of this lesson
    if not doc:
        st.error("Failed to split the pdf")
        st.stop()

    #embeddings step
    search_index = FAISS.from_documents(doc, OpenAIEmbeddings())

    #recursive embeddings step
    #for doc_chunk in doc:
    #    search_index = FAISS.from_documents(doc_chunk, OpenAIEmbeddings())
    #    time.sleep(2)  # Add delay here
    
    return search_index


##HERE IT GOES THE MAIN FUNCTION
def main():
# st.markdown can be used if you want to modify the style of the html page
#not required. this is just an example that removes the hamberger menu which has mostly dev specific options 
#and the footer which has an annoying streamlit link. In any case, you can modify any page element via this function 
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
#this is the title of the app in the main tab
    st.write(
    f"""
    <div style="display: flex; align-items: center; margin-left: 0;">
        <h1 style="display: inline-block;">有知有行投资问答机器人</h1>
    </div>
    """,
    unsafe_allow_html=True,
        )
        
#Element that allows to upload the pdf          
    #uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

#we need to store the question for the history
    #this initializes saving the question in the session  
    if 'question' not in st.session_state:
        st.session_state.question = ""
    
    #this is done to clear the text box after the question is asked.
    #at first the question is saved into widget (the text box), then 
    #gets copied to question and then widget gets deleted 
    def submit():
        st.session_state.question = st.session_state.widget
        st.session_state.widget = ""
    
    #where the user will actually write the question
    st.text_input("Enter your question:", key = "widget", on_change=submit)

    #we initialize session history. here we will put the history
    if 'history' not in st.session_state:
        #create the history data frame. we care about Q, A, and pdf name
        st.session_state['history'] = pandas.DataFrame(columns=['Question', 'Answer', 'Pdf'])

#Below we define the sidebar UI. this is the title in the sidebar
    st.sidebar.title("Parameters")

#this is the text box that lets the user input their openAI key
    default_api_key = os.getenv('OPENAI_API_KEY_SC')

    logging.basicConfig(level=logging.INFO)
    
    if default_api_key is not None:
        logging.info("API Key found.")
    else:
        logging.warning("API Key NOT found.")

    # openai_api_key = st.sidebar.text_input('Please enter your OpenAI API key', 
    #                                        # this is the default value. you can change it to your key
    #                                        value=default_api_key,
    #                                        type = "password", 
    #                                        placeholder="It begins with sk-")
    openai_api_key = default_api_key

#these allow to choose chunk size and overlap 
    chunk_size = st.sidebar.number_input('Please enter the chunk size (number of characters)', value=800)
    overlap_size = st.sidebar.number_input('Please enter the overlap between chunks (number of characters)', value=100)

# #Logic to upload/update pdf.
#     #if no file has been uploaded, print a warning
#     if not uploaded_file:
#        st.warning("Please upload a pdf") 

#     #If a file has been uploaded
#     else:
#       #check if the file uploaded is not in the session state or it is different from the one currently stored
#       if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file:
#           #if so, it means that it is a new file, so we update the file in the session state
#           st.session_state.last_uploaded_file = uploaded_file

#       #actually load the document into the app
#       loaded_text = load_pdf(uploaded_file)

    #Logic to read pdf.
    # set the path to the pdf file
    #pdf_path = 'data/youzhiyouxing_test_2.pdf'
    pdf_path = 'data/youzhiyouxing_v1.0.pdf' 

    # check if the file exists
    if not os.path.exists(pdf_path):
      st.error("Specified PDF file not found.")
      st.stop()

    # read the pdf file
    with open(pdf_path, 'rb') as file:
      loaded_text = load_pdf(file)
      
#check if the api key has been provided and/or needs to be updated. similar logic as for uploaded_file
      #if a key has been provided
      if openai_api_key:
         #if there is no api key saved in the session or the key has changed
         if 'openai_api_key' not in st.session_state or st.session_state.openai_api_key != openai_api_key:
            #add it as an environment variable
            st.session_state.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
      #else tell the user to add one    
      else:
        st.warning("Please add a key")
        return
      
#create and store embeddings based on the uploaded file and the input values of chunk/overlap
      search_index = retriever_function(loaded_text=loaded_text, 
                                         chunk_size = chunk_size, 
                                         overlap_size = overlap_size
                                         )
     
#call OpenAI      
      #define OpenAI model
      qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0), 
                                       retriever=search_index.as_retriever(),
                                       chain_type="stuff")

      # if the user asked a question                                 
      if st.session_state.question:
        #save question to user_question
        user_question = st.session_state.question
        #run the model
        answer = qa.run(user_question)
        #and print the answer
        st.write(user_question, ":", answer)
        #delete the question from the session 
        st.session_state.question = ""
        #append to question history
        #st.session_state['history'].loc[len(st.session_state['history'])] = [user_question, answer, uploaded_file.name]
        st.session_state['history'].loc[len(st.session_state['history'])] = [user_question, answer, pdf_path]

    
    #print history if there was anything in the history prior to the current question. 
    if len(st.session_state['history'])>1:
        #as an example we print just the previous 3 Q&A (last 3 rows in the dataset after removing the last one)
           print_history = st.session_state['history'][:-1].tail(3)
           #add history header
           st.markdown(
                f"""
                    <br><br><br>
                    <p style="font-size: 30px; text-align:center"><b>HISTORY</b></p>
                """,
                unsafe_allow_html=True,
              )
           #for loop reversed order so we have the most recent on top
           for index, row in print_history[::-1].iterrows():
              #print Q, A and pdf name
              st.markdown(
                f"""
                    <br><hr><br>
                    <p><b>{row['Question']}</b>:<br></p>
                    <p>{row['Answer']} </p>
                    <p style="font-size: 14px;">{row['Pdf']}</p>
                """,
                unsafe_allow_html=True,
              )


if __name__ == "__main__":
    main()  
