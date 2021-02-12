from PIL import Image
import streamlit as st
import booste

from session_state import SessionState, get_state

# Unfortunately Streamlit sharing does not allow to hide enviroment variables yet.
# Do not copy this API key, go to https://www.booste.io/ and get your own, it is free!
BOOSTE_API_KEY = "3818ba84-3526-4029-9dc8-ef3038697ea2"


task_name: str = st.sidebar.radio("Task", options=["Image classification", "Image ranking", "Prompt ranking"])

st.markdown("# CLIP playground")
st.markdown("### Try OpenAI's CLIP model in your browser")
st.markdown(" "); st.markdown(" ")
with st.beta_expander("What is CLIP?"):
    st.markdown("Nice CLIP explaination")
st.markdown(" "); st.markdown(" ")
if task_name == "Image classification":
    session_state = get_state()
    uploaded_image = st.file_uploader("Upload image", type=[".png", ".jpg", ".jpeg"],
                                      accept_multiple_files=False)
    st.markdown("or choose one from")
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        default_image_1 = "https://cdn.pixabay.com/photo/2014/10/13/21/34/clipper-487503_960_720.jpg"
        st.image(default_image_1, use_column_width=True)
        if st.button("Select image 1"):
            session_state.image = default_image_1
    with col2:
        default_image_2 = "https://cdn.pixabay.com/photo/2019/12/17/18/20/peacock-4702197_960_720.jpg"
        st.image(default_image_2, use_column_width=True)
        if st.button("Select image 2"):
            session_state.image = default_image_2
    with col3:
        default_image_3 = "https://cdn.pixabay.com/photo/2016/11/15/16/24/banana-1826760_960_720.jpg"
        st.image(default_image_3, use_column_width=True)
        if st.button("Select image 3"):
            session_state.image = default_image_3
    raw_classes = st.text_input("Enter the classes to chose from separated by a comma."
                                " (f.x. `banana, sailing boat, honesty, apple`)")
    if raw_classes:
        session_state.processed_classes = raw_classes.split(",")
        input_prompts = ["A picture of a " + class_name for class_name in session_state.processed_classes]

col1, col2 = st.beta_columns([2, 1])
with col1:
    st.markdown("Image to classify")
    if session_state.image is not None:
        st.image(session_state.image, use_column_width=True)
    else:
        st.warning("Select an image")

with col2:
    st.markdown("Classes to choose from")
    if session_state.processed_classes is not None:
        for class_name in session_state.processed_classes:
            st.write(class_name)
    else:
        st.warning("Enter the classes to classify from")

# Possible way of customize this https://discuss.streamlit.io/t/st-button-in-a-custom-layout/2187/2
if st.button("Predict"):
    with st.spinner("Predicting..."):
        clip_response = booste.clip(BOOSTE_API_KEY,
                        prompts=input_prompts,
                        images=[session_state.image],
                        pretty_print=True)
        st.write(clip_response)


session_state.sync()



