import streamlit as st


# --- PAGE SETUP ---

dashpage = st.Page(
    "views/sales_dashboard.py",
    title="AI SALES DASHBOARD",
    icon=":material/bar_chart:",
    default=True,
)
chatpage = st.Page(
    "views/chatbot.py",
    title="AI CHAT BOT",
    icon=":material/smart_toy:",
)


# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
# pg = st.navigation(pages=[about_page, project_1_page, project_2_page])

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Toolbar": [dashpage, chatpage],
    }
)


st.logo("assets/midhun.png")

# --- RUN NAVIGATION ---
pg.run()