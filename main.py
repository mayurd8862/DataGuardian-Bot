from langchain.chat_models import ChatOpenAI
import streamlit as st
from streamlit_chat import message
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
import io
import tempfile
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate



# Load PDF function
def load_pdf(file_path):
    loaders = [
        PyPDFLoader(file_path)
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    return splits

# Create Chroma database function
def create_chroma_database(splits):
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = 'docs/chroma/'
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )
    return vectordb

# Check if session state variables exist, otherwise initialize them
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

st.title("🤖💬DataGuardian Bot")

# File uploader widget
file = st.file_uploader("Upload PDF File", type=["pdf"])

if file is not None:
    # Save the uploaded file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(file.read())
    file_path = temp_file.name
    temp_file.close()

    # Process the uploaded PDF file
    splits = load_pdf(file_path)
    vectordb = create_chroma_database(splits)

    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="")

    
# Build prompt
    
    template = """Use the following pieces of context to answer the question at the end. If you don't know the  answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the  answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# Run chain




    # Container for chat history
    response_container = st.container()
    # Container for text box
    textcontainer = st.container()

    with textcontainer:
        query = st.text_input("Query: ", key="input")
        if query:
            with st.spinner("typing..."):

                qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
       
            result = qa_chain({"query": query})
            response = result["result"]

            st.session_state.requests.append(query)
            st.session_state.responses.append(response) 

    with response_container:
        if st.session_state['responses']:
            # Display chat history
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i], key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
