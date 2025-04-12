import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool, AgentType, load_tools
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Load your OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="📄 PDF RAG Agent Chatbot", layout="wide")
st.title("📄 Ask Your PDF with Agent Intelligence")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Step 1: Extract text
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    st.success("✅ PDF text extracted.")

    # Step 2: Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    st.info(f"🔹 Document split into {len(chunks)} chunks.")

    # Step 3: Embedding + FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    with st.spinner("🔍 Generating vector embeddings..."):
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        st.success("📈 Vector store created!")

    retriever = vector_store.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search_tool",
        description="Use this to search for answers from the uploaded NEET PDF"
    )

    # Step 4: Define tools (retriever + calculator)
    tools = [retriever_tool]
    tools += load_tools(["llm-math"], llm=ChatOpenAI(openai_api_key=openai_api_key))

    # Step 5: Ask a question
    st.header("🔎 Ask a Question (Search + Math + Memory)")
    question = st.text_input("Type your question here...")

    # Step 6: Memory Setup
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if question and openai_api_key:
        with st.spinner("🤖 Agent is thinking..."):
            agent = initialize_agent(
                tools=tools,
                llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
                agent=AgentType.OPENAI_FUNCTIONS,
                memory=memory,
                verbose=True
            )

            answer = agent.run(question)
            st.success("✅ Answer:")
            st.markdown(answer)
