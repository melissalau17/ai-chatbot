import streamlit as st
from openai import OpenAI
import google.generativeai as genai

# API Configuration
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Sidebar Settings
st.sidebar.title("Chatbot Settings")

selected_model = st.sidebar.selectbox(
    "Choose a model:",
    ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gemini-1.5-flash"],
)
st.session_state["selected_model"] = selected_model

temperature = st.sidebar.slider("Temperature", 0.01, 5.00, 0.7, 0.01)
top_p = st.sidebar.slider("Top P", 0.01, 1.00, 0.9, 0.01)
top_k = st.sidebar.slider("Top K (Gemini only)", 1, 100, 40, 1)
max_tokens = st.sidebar.slider("Max Output Tokens", 64, 4096, 512, 1)

# Chat UI
st.title("Simple Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Response Generator
def generate_response(prompt: str):
    if "gemini" in st.session_state["selected_model"]:
        model = genai.GenerativeModel(st.session_state["selected_model"])
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text

    else:  # OpenAI
        stream = client.chat.completions.create(
            model=st.session_state["selected_model"],
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages] + [
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
        return st.write_stream(stream)

# Summarize Chat
def summarize_chat():
    if not st.session_state.messages:
        return "No chat history to summarize."

    chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
    prompt = f"Summarize the following conversation briefly:\n\n{chat_text}"

    return generate_response(prompt)


if st.sidebar.button("Summarize Chat"):
    with st.sidebar:
        st.markdown("**Chat Summary:**")
        summary = summarize_chat()
        st.success(summary)

# Chat Input
if prompt := st.chat_input("What is up?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        response = generate_response(prompt)
        st.markdown(response)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
