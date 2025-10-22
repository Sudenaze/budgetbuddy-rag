# app.py — BudgetBuddy (RAG + Gemini)
from __future__ import annotations
import os, re, json
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- RAG deps ---
import chromadb
from sentence_transformers import SentenceTransformer

# --- LLM (Gemini) ---
from dotenv import load_dotenv
import google.generativeai as genai


# =========================
# Setup
# =========================
st.set_page_config(
    page_title="BudgetBuddy 💸",
    page_icon="💸",
    layout="wide",               # geniş ekran
    initial_sidebar_state="collapsed",
)
load_dotenv()
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)
TIPS_PATH      = DATA_DIR / "tips.csv"
EXPENSES_PATH  = DATA_DIR / "expenses.json"
CHROMA_DIR     = DATA_DIR / "chroma_db"
COLLECTION     = "bb_tips"

# LLM
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    gemini_model = genai.GenerativeModel(gemini-1.5-flash-8b)
else:
    gemini_model = None

st.info(
    f"LLM status: {'✅ Gemini connected' if gemini_model else '⚠️ No API key (fallback mode)'} "
    f"{'(key ' + GEMINI_KEY[:6] + '…)' if GEMINI_KEY else ''}"
)


# =========================
# Data helpers
# =========================
def load_tips() -> pd.DataFrame:
    if not TIPS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(TIPS_PATH)
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
        df["id"] = df["id"].apply(lambda x: f"T{int(x):03d}")
    return df

def load_expenses() -> pd.DataFrame:
    if not EXPENSES_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "amount", "category", "note"])
    try:
        data = json.load(open(EXPENSES_PATH, "r", encoding="utf-8"))
        df = pd.DataFrame(data)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame(columns=["timestamp", "amount", "category", "note"])

def append_expense(amount: float, category: str, note: str = ""):
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "amount": float(amount),
        "category": category,
        "note": note.strip(),
    }
    data = []
    if EXPENSES_PATH.exists():
        try:
            data = json.load(open(EXPENSES_PATH, "r", encoding="utf-8"))
        except Exception:
            data = []
    data.append(row)
    json.dump(data, open(EXPENSES_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# =========================
# Light NLP (kural tabanlı)
# =========================
CATEGORY_KEYWORDS = {
    "coffee": ["kahve", "coffee"],
    "food": ["yemek", "döner", "pizza", "burger", "restoran", "food", "lunch", "dinner"],
    "transport": ["minibüs", "otobüs", "dolmuş", "metro", "taksi", "uber", "ulaşım", "transport", "bus", "taxi"],
    "entertainment": ["sinema", "film", "oyun", "konser", "eğlence", "netflix", "spotify", "cinema", "movie"],
    "utilities": ["fatura", "elektrik", "su", "doğalgaz", "internet", "abonelik", "utilities", "bill"],
    "shopping": ["alışveriş", "market", "kıyafet", "giyim", "ayakkabı", "shopping", "clothes"],
    "other": [],
}
def guess_category(text: str) -> str:
    t = text.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(k in t for k in kws):
            return cat
    return "other"
def extract_amount(text: str):
    m = re.search(r"(\d+[.,]?\d*)\s*(tl|₺)?", text.lower())
    if not m: return None
    num = m.group(1).replace(",", ".")
    try: return float(num)
    except: return None


# =========================
# RAG: build + semantic search
# =========================
@st.cache_resource(show_spinner=False)
def _embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # hafif & hızlı

@st.cache_resource(show_spinner=True)
def _chroma_collection(tips_df: pd.DataFrame):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    if col.count() == 0 and not tips_df.empty:
        model = _embedder()
        texts = tips_df["advice"].astype(str).tolist()
        ids = tips_df["id"].astype(str).tolist()
        metas = tips_df.fillna("").to_dict(orient="records")
        emb = model.encode(texts, batch_size=64, normalize_embeddings=True).tolist()
        col.add(documents=texts, embeddings=emb, ids=ids, metadatas=metas)
    return col

def rag_search(query: str, tips_df: pd.DataFrame, k: int = 6) -> list[dict]:
    if tips_df.empty:
        return []
    col = _chroma_collection(tips_df)
    q_emb = _embedder().encode([query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=k)
    out = []
    for i in range(len(res["ids"][0])):
        out.append({
            "id": res["ids"][0][i],
            "advice": res["documents"][0][i],
            "meta": res["metadatas"][0][i],
            "score": 1 - res["distances"][0][i],
        })
    return out


# =========================
# LLM: compose answer
# =========================
def build_llm_prompt(user_msg: str, retrieved: list[dict], weekly_summary: str) -> str:
    tips_bullets = "\n".join([f"- {t['advice']}" for t in retrieved]) if retrieved else "- (no retrieved tips)"
    return f"""
You are BudgetBuddy, a warm, clear, practical personal finance assistant.
User message: ```{user_msg}```

Context:
- User's last 7-day spending summary: {weekly_summary}
- Retrieved financial tips from our knowledge base:
{tips_bullets}

Instructions:
- Give 3–5 concise, actionable suggestions tailored to the user's message and weekly summary.
- If user mentions a spend (like coffee 80 TL), acknowledge it briefly and give 1 habit change idea.
- Keep tone friendly, motivating, and specific; avoid generic filler.
- Write in English. Use bullet points.
"""

def llm_answer(user_msg: str, retrieved: list[dict], exp: pd.DataFrame) -> str:
    # build a simple last-7-days summary
    if exp.empty:
        weekly = "no expenses logged yet."
    else:
        last7 = exp[exp["timestamp"] >= (pd.Timestamp.now() - pd.Timedelta(days=7))]
        if last7.empty:
            weekly = "no expenses in the last 7 days."
        else:
            top_cat = last7.groupby("category")["amount"].sum().sort_values(ascending=False).index[0]
            spent = last7["amount"].sum()
            weekly = f"spent ₺{spent:,.0f} total; top category = {top_cat}"

    if gemini_model is None:
        # Fallback: rule-based short answer
        bullets = "\n".join([f"- {t['advice']}" for t in retrieved[:3]]) if retrieved else "- Try tracking a few expenses first."
        return f"(fallback)\nHere are some suggestions:\n{bullets}"

    prompt = build_llm_prompt(user_msg, retrieved, weekly)
    try:
        resp = gemini_model.generate_content(prompt)
        return resp.text.strip() if hasattr(resp, "text") else "Sorry, I couldn't generate a response."
    except Exception as e:
        return f"(LLM error) {e}\n\n" + \
               ("\n".join([f"- {t['advice']}" for t in retrieved[:3]]) if retrieved else "- Try logging some expenses.")


# =========================
# UI
# =========================
tips_df = load_tips()
if not tips_df.empty:
    st.success(f"Tips dataset loaded ✅ (rows: {len(tips_df)})")
else:
    st.error("Couldn't find data/tips.csv — please add it.")

tab1, tab2, tab3 = st.tabs(["➕ Add Expense", "💬 Chat (RAG+LLM)", "📊 Report"])

# --- Add Expense ---
with tab1:
    st.subheader("➕ Add Expense")
    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input("Amount (₺)", min_value=0.0, step=10.0, value=60.0, key="amount_input")
    with c2:
        category = st.selectbox("Category", list(CATEGORY_KEYWORDS.keys()), index=0, key="category_select")
    note = st.text_input("Note (optional)", key="note_input")
    if st.button("Save", key="save_btn"):
        if amount <= 0:
            st.warning("Amount must be greater than 0.")
        else:
            append_expense(amount, category, note)
            st.success("Saved ✅")
            st.rerun()

    exp = load_expenses()
    if not exp.empty:
        st.markdown("**Recent entries**")
        st.dataframe(exp.sort_values("timestamp", ascending=False).head(8), use_container_width=True)
    else:
        st.info("No expenses yet. Add your first one above.")

# --- Chat (RAG + LLM) ---
with tab2:
    st.subheader("💬 Chat (RAG + Gemini)")
    st.caption("Type naturally, e.g. *“I spent 80 TL on coffee”* or *“shopping savings tips please”*.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi! I can log your spending and suggest smarter ways to save."}
        ]
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_msg = st.chat_input("Your message…")
    if user_msg:
        st.session_state["messages"].append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)

        # auto-log if a spend is detected
        amt = extract_amount(user_msg)
        cat = guess_category(user_msg)
        logged = ""
        if amt is not None and cat:
            append_expense(amt, cat, note=user_msg)
            logged = f"Logged ₺{amt:,.0f} for '{cat}'. "

        # RAG retrieve + LLM generate
        retrieved = rag_search(user_msg, tips_df, k=6) if not tips_df.empty else []
        exp = load_expenses()
        answer = llm_answer(user_msg, retrieved, exp)

        final_text = (logged + answer).strip()
        st.session_state["messages"].append({"role": "assistant", "content": final_text})
        with st.chat_message("assistant"):
            st.write(final_text)

# --- Report ---
with tab3:
    st.subheader("📊 Report")
    exp = load_expenses()
    if exp.empty:
        st.info("No data yet. Add some expenses to see reports.")
    else:
        total = exp["amount"].sum()
        this_month = exp[exp["timestamp"].dt.to_period("M") == pd.Timestamp.now().to_period("M")]
        monthly_total = this_month["amount"].sum()

        c1, c2 = st.columns(2)
        c1.metric("Total spent", f"₺{total:,.0f}")
        c2.metric("This month", f"₺{monthly_total:,.0f}")

        by_cat = exp.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
        st.markdown("**By category**")
        st.dataframe(by_cat, use_container_width=True)

        fig = plt.figure()
        plt.bar(by_cat["category"], by_cat["amount"])
        plt.title("Spending by Category")
        plt.xlabel("Category")
        plt.ylabel("₺")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)

st.markdown("---")
st.caption("v1.0 — RAG + Gemini · logs spending automatically when you write amounts (e.g., '80 TL coffee').")
