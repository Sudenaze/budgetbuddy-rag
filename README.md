# budgetbuddy-rag
RAG-based financial awareness chatbot
This is a **RAG (Retrieval-Augmented Generation)** based financial awareness chatbot project designed for Akbank GenAI Bootcamp.

About the Project
BudgetBuddy was developed to address the challenge young people face when planning their expenses.

Users write messages in natural language:

> “I spent 80 TL on coffee today.”

> “I spend a lot on food, what can I do?”
>
> The chatbot analyzes the message, records the spending, and provides appropriate suggestions from its own **financial advice knowledge base**. This helps the user gain budget awareness and learn how to save.
>
> Technologies Used
> **Python** – Ana geliştirme dili  
- **Streamlit** – Web arayüzü (chatbot ekranı)  
- **Gemini API / OpenAI** – Metin üretimi (LLM)  
- **Sentence Transformers** – Embedding oluşturmak için  
- **Chroma DB** – Vektör veritabanı (finansal ipuçlarını saklamak için)  
- **Matplotlib** – Haftalık rapor/grafik için  
- **JSON** – Kullanıcı harcamalarını kaydetmek için

Project Structure
BudgetBuddy/
│
├── app.py # Main application file (Streamlit)
├── data/
│ └── tips.csv # Financial tips dataset
├── requirements.txt # Required libraries
├── README.md # Project description
└── .env (optional) # API keys (Gemini etc.)







