import streamlit as st
import pandas as pd
from pathlib import Path

# --- Streamlit basic setup ---
st.set_page_config(page_title="BudgetBuddy 💸", page_icon="💸", layout="centered")

DATA_PATH = Path("data") / "tips.csv"

# --- Header ---
st.title("💸 BudgetBuddy")
st.caption("RAG-based financial awareness assistant (MVP setup).")

# --- Load tips.csv ---
if DATA_PATH.exists():
    tips_df = pd.read_csv(DATA_PATH)
    st.success(f"Tips dataset loaded ✅ (rows: {len(tips_df)})")
else:
    tips_df = pd.DataFrame()
    st.error("Couldn't find data/tips.csv — please add it.")

# --- Tabs setup ---
tab1, tab2, tab3 = st.tabs(["➕ Add Expense", "💡 Tips (RAG)", "📊 Report"])

with tab1:
    st.subheader("➕ Add Expense (placeholder)")
    st.write("This section will record spending in later versions (JSON log).")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount (₺)", min_value=0.0, step=10.0)
    with col2:
        category = st.selectbox("Category", ["coffee", "food", "transport", "entertainment", "other"])

    note = st.text_input("Note (optional)")
    st.button("Save (coming soon)")

with tab2:
    st.subheader("💡 Financial Tips (Preview Mode)")
    st.write("RAG system will retrieve personalized tips here. For now, preview a few rows.")
    if not tips_df.empty:
        st.dataframe(tips_df.head(5))
        st.info("RAG retrieval and generation will be added next.")
    else:
        st.warning("Tips dataset not loaded.")

with tab3:
    st.subheader("📊 Reports (placeholder)")
    st.write("Summary and charts will appear here soon.")
    st.progress(0)

st.markdown("---")
st.caption("Prototype v0.1 — next: JSON logging, embeddings, and RAG answers.")
