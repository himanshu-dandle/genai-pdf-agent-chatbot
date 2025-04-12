import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Load your OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="📄 PDF RAG Chatbot", layout="wide")
st.title("📄 Ask Your PDF (RAG + Agent Powered)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Step 1: Extract text from PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    st.success("✅ PDF text extracted.")

    # Step 2: Chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    st.info(f"🔹 Document split into {len(chunks)} chunks.")

    # Step 3: Generate embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    with st.spinner("🔍 Generating vector embeddings..."):
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        st.success("📈 Vector store created!")

    # Optional: Save vectorstore to disk
    vector_store.save_local("faiss_index")

    # Step 4: Ask a question
    st.header("🔎 Ask a Question from the PDF")
    question = st.text_input("Type your question here...")

    if question and openai_api_key:
        with st.spinner("🤖 Thinking..."):
            # Step 5: Retrieve relevant chunks
            docs = vector_store.similarity_search(question, k=3)

            # Combine chunks as context
            context = "\n\n".join([doc.page_content for doc in docs])

            # Step 6: Generate GPT-based answer
            from langchain.chat_models import ChatOpenAI
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain

            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""
You are a helpful NEET assistant AI.

Use the following context to answer the question accurately.
If you don't know the answer, say "Sorry, I couldn't find that in the document."

Context:
{context}

Question: {question}
Answer:
"""
            )

            llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.1)
            chain = LLMChain(llm=llm, prompt=prompt)

            answer = chain.run(context=context, question=question)

            st.success("✅ Answer:")
            st.markdown(answer)
