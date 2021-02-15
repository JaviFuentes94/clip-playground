from typing import Optional, List

from PIL import Image
import streamlit as st
import booste

from session_state import SessionState, get_state

# Unfortunately Streamlit sharing does not allow to hide enviroment variables yet.
# Do not copy this API key, go to https://www.booste.io/ and get your own, it is free!
BOOSTE_API_KEY = "3818ba84-3526-4029-9dc8-ef3038697ea2"


class Sections:
    @staticmethod
    def header():
        st.markdown("# CLIP playground")
        st.markdown("### Try OpenAI's CLIP model in your browser")
        st.markdown(" ");
        st.markdown(" ")
        with st.beta_expander("What is CLIP?"):
            st.markdown("Nice CLIP explaination")
        st.markdown(" ");
        st.markdown(" ")

    @staticmethod
    def image_uploader(accept_multiple_files: bool) -> Optional[List[str]]:
        uploaded_image = st.file_uploader("Upload image", type=[".png", ".jpg", ".jpeg"],
                                          accept_multiple_files=accept_multiple_files)

    @staticmethod
    def image_picker(state: SessionState):
        col1, col2, col3 = st.beta_columns(3)
        with col1:
            default_image_1 = "https://cdn.pixabay.com/photo/2014/10/13/21/34/clipper-487503_960_720.jpg"
            st.image(default_image_1, use_column_width=True)
            if st.button("Select image 1"):
                state.image = default_image_1
        with col2:
            default_image_2 = "https://cdn.pixabay.com/photo/2019/12/17/18/20/peacock-4702197_960_720.jpg"
            st.image(default_image_2, use_column_width=True)
            if st.button("Select image 2"):
                state.image = default_image_2
        with col3:
            default_image_3 = "https://cdn.pixabay.com/photo/2016/11/15/16/24/banana-1826760_960_720.jpg"
            st.image(default_image_3, use_column_width=True)
            if st.button("Select image 3"):
                state.image = default_image_3

    @staticmethod
    def prompts_input(state: SessionState, input_label: str, prompt_prefix: str = ''):
        raw_classes = st.text_input(input_label)
        if raw_classes:
            state.prompts = [prompt_prefix + class_name for class_name in raw_classes.split(";") if len(class_name) > 1]
            state.prompt_prefix = prompt_prefix

    @staticmethod
    def input_preview(state: SessionState):
        col1, col2 = st.beta_columns([2, 1])
        with col1:
            st.markdown("Image to classify")
            if state.image is not None:
                st.image(state.image, use_column_width=True)
            else:
                st.warning("Select an image")

        with col2:
            st.markdown("Labels to choose from")
            if state.processed_classes is not None:
                for prompt in state.prompts:
                    st.write(prompt[len(state.prompt_prefix):])
            else:
                st.warning("Enter the classes to classify from")

    @staticmethod
    def classification_output(state: SessionState):
        # Possible way of customize this https://discuss.streamlit.io/t/st-button-in-a-custom-layout/2187/2
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                clip_response = booste.clip(BOOSTE_API_KEY,
                                            prompts=state.prompts,
                                            images=[state.image],
                                            pretty_print=True)
                st.markdown("### Results")
                simplified_clip_results = [(prompt[len(state.prompt_prefix):],
                                            list(results.values())[0]["probabilityRelativeToPrompts"])
                                           for prompt, results in clip_response.items()]
                simplified_clip_results = sorted(simplified_clip_results, key=lambda x: x[1], reverse=True)

                for prompt, probability in simplified_clip_results:
                    percentage_prob = int(probability * 100)
                    st.markdown(
                        f"### ![prob](https://progress-bar.dev/{percentage_prob}/?width=200) &nbsp &nbsp {prompt}")
                st.write(clip_response)


task_name: str = st.sidebar.radio("Task", options=["Image classification", "Image ranking", "Prompt ranking"])
session_state = get_state()
if task_name == "Image classification":
    Sections.header()
    Sections.image_uploader(accept_multiple_files=False)
    st.markdown("or choose one from")
    Sections.image_picker(session_state)
    input_label = "Enter the classes to chose from separated by a semi-colon. (f.x. `banana; boat; honesty; apple`)"
    Sections.prompts_input(session_state, input_label, prompt_prefix='A picture of a ')
    Sections.input_preview(session_state)
    Sections.classification_output(session_state)
elif task_name == "Prompt ranking":
    Sections.header()
    Sections.image_uploader(accept_multiple_files=False)
    st.markdown("or choose one from")
    Sections.image_picker(session_state)
    input_label = "Enter the prompts to choose from separated by a semi-colon. " \
                  "(f.x. `An image that inspires; A feeling of loneliness; joyful and young; apple`)"
    Sections.prompts_input(session_state, input_label)
    Sections.input_preview(session_state)
    Sections.classification_output(session_state)
elif task_name == "Image ranking":
    Sections.header()
    Sections.image_uploader(accept_multiple_files=True)
    st.markdown("or use random dataset")
    Sections.image_picker(session_state)



session_state.sync()



