✦ My RAG AI

«An AI-powered study assistant that turns your PDFs into an interactive learning workspace.»

"🚀 Live Demo" (https://rag-ai-assistance.streamlit.app/)

⚡ What it does

Upload your study PDFs and chat with them using Retrieval-Augmented Generation (RAG).

- 💬 Ask AI — Ask questions grounded in your uploaded documents
- 📝 Quiz Generator — Generate quizzes from your study material
- 🧠 Flashcards — Create interactive revision flashcards
- 📖 Smart Summary — Get focused summaries of your PDFs
- 🎯 Important Questions — Find questions worth preparing
- 📚 Source Citations — See which document/page supports an answer

🧠 RAG Pipeline

PDF Upload
    ↓
Text Extraction
    ↓
Chunking
    ↓
HuggingFace Embeddings
    ↓
Chroma Vector Database
    ↓
Relevant Context Retrieval
    ↓
Gemini AI
    ↓
Grounded Answer

🛠️ Tech Stack

Python · Streamlit · Google Gemini · LangChain · ChromaDB · HuggingFace Embeddings

📁 Project Structure

My-RAG-AI/
├── app.py
├── test.py
├── requirements.txt
├── README.md
└── .gitignore

🚀 Run Locally

git clone <your-repository-url>
cd My-RAG-AI
pip install -r requirements.txt
streamlit run app.py

Set your Gemini API key as:

GEMINI_API_KEY=your_api_key

🌐 Live App

Try My RAG AI:
https://rag-ai-assistance.streamlit.app/

---

Built with ✦ by Swastik