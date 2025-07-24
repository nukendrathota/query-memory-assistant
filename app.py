# app.py
import streamlit as st
from db_utils import get_embedding, find_similar_inference, generate_response, save_inference

st.set_page_config(page_title="Mini AI Assistant", layout="centered")

st.title("🧠 Ask the Assistant")
user_input = st.text_input("Type your question here:")

if st.button("Submit") and user_input.strip():
    with st.spinner("Generating embedding..."):
        embedding = get_embedding(user_input)

    if not embedding:
        st.error("Failed to generate embedding. Please check your internet/API connection.")
    else:
        with st.spinner("Checking for similar cached responses..."):
            match = find_similar_inference(embedding)

        if match and match["distance"] < 0.1:
            st.success("✅ Found cached answer:")
            st.write(match["output_text"])
            st.caption(f"🔁 Similarity distance: {match['distance']:.4f}")
        else:
            with st.spinner("No match found. Generating fresh answer..."):
                response = generate_response(user_input)

            st.success("🤖 Fresh AI answer:")
            st.write(response)
            save_inference(user_input, "gpt-3.5-turbo", response, embedding)
