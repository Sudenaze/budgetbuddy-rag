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

## 🛠️ Technologies Used

| Technology | Description |
|-------------|-------------|
| **Python** | The core programming language used to build and run the chatbot logic. |
| **Streamlit** | Framework for creating an interactive and user-friendly web interface. |
| **Gemini API / OpenAI API** | Large Language Models (LLMs) responsible for generating personalized financial advice. |
| **Sentence Transformers** | Converts text data (financial tips) into numerical embeddings for semantic similarity search. |
| **Chroma DB** | Vector database used to store embeddings and retrieve the most relevant financial advice snippets. |
| **Matplotlib** | Generates visual budget summaries and spending charts within the Streamlit dashboard. |
| **JSON** | Lightweight format used to store user spending records and local application data. |


```##Project Structure
BudgetBuddy/
│
├── app.py # Main application file (Streamlit)
├── data/
│ └── tips.csv # Financial tips dataset
├── requirements.txt # Required libraries
├── README.md # Project description
└── .env (optional) # API keys (Gemini etc.)
── expenses.json # 
└── goals.json #
```






