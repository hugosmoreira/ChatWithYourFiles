# Your original imports
import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from load_docs import load_docs
import streamlit as st
from streamlit_chat import message

# Revised import for Chroma client
import chromadb

# Load environment variables
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# Using OpenAI Chat API
llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0, model=llm_model)

# Load documents (assuming this function is defined in another file, unchanged)
documents = load_docs()
chat_history = []

# Split the data into chunks
text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=10)
docs = text_splitter.split_documents(documents)

# Create our vector db using Chroma - revised client creation
client = chromadb.PersistentClient(path="./data")

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    client=client  # Using the new client
)

# Define the QA Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

# Streamlit front-end
st.title("Docs Hugo's QA Bot using Langchain")
st.header("Ask anything about your documents... 🤖")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
    
if 'past' not in st.session_state:
    st.session_state['past'] = []
    
def get_query():
    input_text = st.chat_input("Ask a question about your documents...")
    return input_text

# Retrieve the user input
user_input = get_query()
if user_input:
    result = qa_chain({'question': user_input, 'chat_history': chat_history})
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result['answer'])
    
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['generated'][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i)+ '_user')
