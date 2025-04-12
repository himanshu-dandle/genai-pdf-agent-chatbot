import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool, AgentType, load_tools
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.memory import ConversationBufferMemory
from presidio_analyzer import AnalyzerEngine
import re
import os
from dotenv import load_dotenv

# Load OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit app setup
st.set_page_config(page_title="📄 PDF RAG Agent Chatbot", layout="wide")
st.title("📄 Ask Your PDF with Agent Intelligence")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# PII protection toggle
pii_mode = st.radio(
    "🛡️ Choose document type for privacy:",
    ["🔓 Public (No redaction)", "🔒 Confidential (Redact PII)"]
)

if uploaded_file:
    # Step 1: Extract text
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    st.success("✅ PDF text extracted.")

    # Step 2: Detect and redact PII if needed
    if pii_mode == "🔒 Confidential (Redact PII)":
        analyzer = AnalyzerEngine()
        pii_results = analyzer.analyze(text=text, language='en')

        # Filter only key sensitive entity types
        entities_to_redact = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "NRP", "CREDIT_CARD"]
        redacted_count = 0

        # Sort PII results in reverse order to avoid index shift during replacement
        pii_results_sorted = sorted(pii_results, key=lambda x: x.start, reverse=True)

        for pii in pii_results_sorted:
            if pii.entity_type in entities_to_redact:
                text = text[:pii.start] + "[REDACTED]" + text[pii.end:]
                redacted_count += 1

        # Add regex-based redaction for salary and CTC-like phrases
        salary_patterns = [r'salary[:\s]+[\u20B9Rs.\s]*[\d,]+', r'ctc[:\s]+[\u20B9Rs.\s]*[\d,]+']
        for pattern in salary_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            for match in matches:
                text = text.replace(match, "[REDACTED]")
                redacted_count += 1

        if redacted_count > 0:
            st.info(f"🔐 {redacted_count} PII elements redacted.")
        else:
            st.info("✅ No sensitive PII found in this document.")
    else:
        st.info("🔓 No PII redaction applied (Public mode).")

    # Step 3: Chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    st.info(f"🔹 Document split into {len(chunks)} chunks.")

    # Step 4: Generate embeddings and create FAISS vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    with st.spinner("🔍 Generating vector embeddings..."):
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        st.success("📈 Vector store created!")

    # Step 5: Create retriever tool
    retriever = vector_store.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search_tool",
        description="Use this to search for answers from the uploaded PDF"
    )

    # Step 6: Load tools (retriever + calculator)
    tools = [retriever_tool]
    tools += load_tools(["llm-math"], llm=ChatOpenAI(openai_api_key=openai_api_key))

    # Step 7: Setup memory once for the session
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools=tools,
        llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        verbose=True
    )

    # Step 8: User question input
    st.header("🔎 Ask a Question (Search + Math + Memory)")
    question = st.text_input("Type your question here...")

    if question:
        with st.spinner("🤖 Agent is thinking..."):
            answer = agent.run(question)
            st.success("✅ Answer:")
            st.markdown(answer)
