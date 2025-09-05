import os
import time
import tempfile
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
# langchain core classes and utilities
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader



load_dotenv()

st.set_page_config(
    page_title= " 📝 RAG Q&A with PDF uploads and chat history",
    layout = "wide",
    initial_sidebar_state="expanded"
)

st.title("📝 RAG Q&A with PDF uploades and chat history")

st.sidebar.header(" 🛠 Configuration")

st.sidebar.write(
    " - Enter your Groq API KEY \n"
    " - Upload PDFs on the main page \n"
    " - Ask questions aand see chat history"
)

# api key and embedding setup
api_key = st.sidebar.text_input("Groq API Key", type="password")

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN","") #for hugging face embeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# only proceed if the user has entered their Groq_key
if not api_key:
    st.warning( " 🔑 Please enter your Groq API Key in the sidebar to continue ")

# instntiate the GROQ LLM
llm  = ChatGroq(groq_api_key= api_key,model_name="gemma2-9b-it")

uploaded_files = st.file_uploader(
    " 📑 Choose PDF files(s)",
    type ="pdf",
    accept_multiple_files=True
)

all_docs = []

if uploaded_files:
    # show progress spinner while loading
    with st.spinner(" 🔄 Loading and splitting PDFs"):
        for pdf in uploaded_files:
            #write to a temp file so aa PyPDFLoader caan reaad it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.getvalue())
                pdf_path = tmp.name
            
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
    #split doc into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150
    )
    splits = text_splitter.split_documents(all_docs)

    # Build or load the chroma vector store (caching for performance)
    def get_vectorstore(_splits):
        return Chroma.from_documents(
            _splits,
            embeddings,
            persist_directory = "./chroma_index"
        )
    vectorstore = get_vectorstore(splits)
    retriever = vectorstore.as_retriever()

    # build a history aware retriever that uses past chat to refine serches

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the latest user question, decide what to retrieve."),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    # QA Chain "stuff" all retrieved docss into the LLM
    # Prompt engineering
    # must watch prompt engineering free videos by Andrew NG 

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system","You are an assistnt. Use the retrieved context to answer." 
        "If you don't know, say so. keep it under three sentences. \n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

    # session state for chat history
    if "chathistory" not in st.session_state:
      st.session_state.chathistory = {}


    def get_history(session_id:str):
        if session_id not in st.session_state.chathistory:
            st.session_state.chathistory[session_id] = ChatMessageHistory()
        return st.session_state.chathistory[session_id]
    
    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_message_key= "input",
        history_messages_key = "chat_history",
        output_messages_key = "answer"
    )

    #chatui
    session_id = st.text_input(" 🆔 Session ID", value =" default_session")
    user_question = st.chat_input(" ✍🏻 Your Question here......")

    if user_question:
        history = get_history(session_id)
        result = conversational_rag.invoke(
            {"input" : user_question},
            config = {"configurable": {"session_id": session_id}},
        )
        answer = result["answer"]

        # display in streamlit new chat format

        st.chat_message("user").write(user_question)
        st.chat_message("assistant").write(answer)

        with st.expander(" 📕📙📒📗📘 Full Chat history "):
            for msg in history.messages:
                #msg role is typically "human" or "assitant"
                role = getattr(msg, "role", msg.type)
                content = msg.content
                st.write(f" ** {role.title()}: **. {content}")

else:
    # no file is uploaded yet
    st.info("🚨 Upload one or more PDFs above to begin.")