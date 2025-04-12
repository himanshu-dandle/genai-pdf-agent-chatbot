# PDF RAG Chatbot with LangChain Agents

This project is an intelligent PDF assistant built using **LangChain**, **OpenAI**, **FAISS**, and **Streamlit**. It allows you to upload a NEET/NCERT-style PDF and ask questions — the agent will smartly answer using document retrieval, GPT, and even do math!


## Live Demo

👉 [Try the public version](https://genai-pdf-agent-chatbot-wk5cjv9mjsahrjwbxqx8qc.streamlit.app/)

👉 [Try the PII-Safe Version](https://genai-pdf-agent-chatbot.streamlit.app/?app=app_with_pii) *(with salary + personal info redaction)*


---

##  Features

- Upload any NEET-style, textbook, or even confidential PDF
- RAG: Uses FAISS + OpenAI Embeddings for semantic search
- LangChain Agent handles:
  - Retrieval from PDF
  - Math calculations
  - Smart summarization
  - PII-safe responses
- LangChain Memory: Multi-turn conversations with context
- PII Redaction (via Microsoft Presidio + regex)
- Built with: `Streamlit`, `LangChain`, `OpenAI`, `FAISS`, `PyPDF2`

---

##  Demo

![screenshot](screenshot.png) 


---

##  Installation

```
git clone https://github.com/your-username/pdf-rag-agent-chatbot.git
cd pdf-rag-agent-chatbot
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt


Create a .env file:

OPENAI_API_KEY=your_openai_key

 Run the App
 streamlit run app.py
# or for PII-safe version

streamlit run app_with_pii.py

Then open http://localhost:8501 in your browser.
```
## Tech Stack

| Tool        | Purpose                        |
|-------------|---------------------------------|
| Streamlit   | UI & Chat Interface            |
| LangChain   | RAG, Agent, Memory             |
| OpenAI      | GPT + Embeddings               |
| FAISS       | Fast Vector DB for Retrieval   |
| Presidio    | PII detection (via Microsoft)  |
| PyPDF2      | PDF Parsing                    |
| dotenv      | Secure API key storage         |



## Ideal For
- NEET / JEE / CBSE PDF Study Assistants
- Smart textbook Q&A apps
- Internal HR chatbots (salary, policy docs)
- GenAI project portfolios

##  License
MIT License. Free for learning, education, and portfolio projects.