import streamlit as st
from dotenv import load_dotenv
import os
import requests

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from tavily import TavilyClient
from langchain.agents import create_agent

# =========================
# 🎨 Page Config
# =========================
st.set_page_config(page_title="City Assistant", page_icon="🌍", layout="centered")


# =========================
# 🌦️ Weather Tool
# =========================
@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"❌ Error: {data.get('message')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"🌦️ Weather in {city}: {desc}, {temp}°C"


# =========================
# 📰 News Tool
# =========================
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""

    response = tavily_client.search(
        query=f"latest news in {city}", search_depth="basic", max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    news_list = []

    for r in results:
        news_list.append(f"📰 {r['title']}\n🔗 {r['url']}")

    return "\n\n".join(news_list)


# =========================
# 🧠 LLM + Agent
# =========================
llm = ChatMistralAI(model="mistral-small-2506")

agent = create_agent(
    llm,
    tools=[get_weather, get_news],
    system_prompt="""
    You are a helpful city assistant.

    IMPORTANT:
    - Always use tools for weather and news queries
    - Never say you don't have access to tools
    - Use tool results to answer clearly
    """,
)

# =========================
# 💬 Session State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# 🖥️ UI Header
# =========================
st.title("🌍 City Assistant")
st.caption("Ask about weather 🌦️ or news 📰")

# =========================
# 💬 Show Chat
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# ✍️ Input
# =========================
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )

            bot_reply = result["messages"][-1].content
            st.markdown(bot_reply)

    # Save response
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
