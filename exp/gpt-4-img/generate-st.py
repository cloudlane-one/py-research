"""Generate images via OpenAI DALL-E 3."""

import os
from io import BytesIO
from typing import cast

import openai
import PIL.Image as img  # noqa: N813
import streamlit as st
from dotenv import load_dotenv
from py_research.ai import generate_image

load_dotenv()  # This loads the contents of the .env file into the environment
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Generate images via OpenAI DALL-E 3")

prompt = st.text_input("Prompt", "A cute raccoon")

if st.button("Generate", type="primary"):
    with st.spinner("Generating..."):
        st.session_state["image"] = generate_image(prompt)

    if st.session_state["image"] is None:
        st.error("Did not receive image in response.")

if "image" in st.session_state:
    image = cast(img.Image, st.session_state["image"])

    st.image(image, caption=prompt)

    img_bytes = BytesIO()
    image.save(img_bytes, format="jpeg")
    st.download_button(
        "Download image",
        img_bytes,
        file_name=f"dalle3-gen-{str(abs(hash(img_bytes.getvalue())))[:8]}.jpg",
    )
