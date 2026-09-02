✦ My RAG AI

«Turn your documents into an intelligent study system.»

An AI-powered Retrieval-Augmented Generation (RAG) study assistant that transforms your PDFs into an interactive knowledge base.

Upload your study material, ask questions, generate quizzes, create flashcards, build revision notes, and prepare important questions — all powered by your own documents.

✨ What You Can Do

💬 Ask AI
Chat with your uploaded documents and get context-grounded answers with source references.

📝 Quiz Generator
Generate MCQ quizzes from your study material with difficulty selection and automatic scoring.

🧠 AI Flashcards
Turn important concepts into interactive revision cards for quick learning.

📖 Smart Summary
Generate quick revision sheets, detailed notes, or exam-focused summaries.

🎯 Important Questions
Generate exam-focused questions and identify the key points required for strong answers.

🔎 Smart Retrieval
Uses semantic search and MMR retrieval to find relevant information from your documents.

---

🧠 RAG Pipeline

PDF Documents
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
HuggingFace Embeddings
      ↓
ChromaDB Vector Store
      ↓
MMR Retrieval
      ↓
Relevant Context
      ↓
Google Gemini
      ↓
Answer + Sources

---

🛠️ Tech Stack

Technology| Role
🐍 Python| Core application
🎈 Streamlit| Web interface
🔗 LangChain| Text processing & retrieval
🗄️ ChromaDB| Vector database
🤗 HuggingFace| Embeddings
✦ Google Gemini| AI generation
📄 PyPDF| PDF extraction

---

🚀 Run Locally

git clone YOUR_REPOSITORY_URL
cd my-rag-ai
pip install -r requirements.txt
streamlit run app.py

Set your Gemini API key as an environment variable:

GEMINI_API_KEY

«🔐 Never commit your API key or other secrets to GitHub.»

---

📂 Project Structure

my-rag-ai/
├── app.py
├── test.py
├── requirements.txt
├── .gitignore
└── README.md

---

🎯 Why I Built This

My goal was to build more than a basic chatbot and gain practical experience with modern AI systems.

This project combines:

- Retrieval-Augmented Generation
- Vector databases
- Text embeddings
- Semantic search
- MMR retrieval
- Prompt engineering
- Generative AI
- AI-powered educational tools

---

🔮 What's Next

- 📌 Persistent knowledge bases
- 📌 Better citation highlighting
- 📌 PDF page previews
- 📌 Subject-wise workspaces
- 📌 Study progress tracking
- 📌 Improved retrieval evaluation

---

👨‍💻 Built By

Swastik

"AI" · "RAG" · "Data Science" · "Intelligent Systems"

---

<p align="center">✦ My RAG AI

Built with Python · Streamlit · LangChain · ChromaDB · HuggingFace · Gemini

</p>