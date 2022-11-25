import requests
import streamlit as st
from PIL import Image

MODELS = {
    "Best Model": "best_model.ckpt",
    "Last Checkpoint": "last.ckpt",
}

st.title("MLOps Challenge Inference Dashboard")

# Model Selection
selected_checkpoint = st.selectbox(
    "Choose your preferred model checkpoint", [i for i in MODELS.keys()]
)
print(selected_checkpoint)

if st.button("Run Inference on Test Images"):

    # Send request
    res = requests.post(f"http://api:8080/{MODELS[selected_checkpoint]}")
    image = Image.open(f"/storage/00001.png")
    st.image(image, width=1000)
