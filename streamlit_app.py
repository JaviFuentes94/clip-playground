import random
import requests

import streamlit as st
from clip_model import ClipModel

from session_state import SessionState, get_state
from images_mocker import ImagesMocker

from PIL import Image

IMAGES_LINKS = ["https://cdn.pixabay.com/photo/2014/10/13/21/34/clipper-487503_960_720.jpg",
                "https://cdn.pixabay.com/photo/2019/09/06/04/25/beach-4455433_960_720.jpg",
                "https://cdn.pixabay.com/photo/2019/11/11/14/30/zebra-4618513_960_720.jpg",
                "https://cdn.pixabay.com/photo/2020/11/04/15/29/coffee-beans-5712780_960_720.jpg",
                "https://cdn.pixabay.com/photo/2020/03/24/20/42/namibia-4965457_960_720.jpg",
                "https://cdn.pixabay.com/photo/2020/08/27/07/31/restaurant-5521372_960_720.jpg",
                "https://cdn.pixabay.com/photo/2020/08/24/21/41/couple-5515141_960_720.jpg",
                "https://cdn.pixabay.com/photo/2020/01/31/07/10/billboards-4807268_960_720.jpg",
                "https://cdn.pixabay.com/photo/2017/07/31/20/48/shell-2560930_960_720.jpg",
                "https://cdn.pixabay.com/photo/2020/08/13/01/29/koala-5483931_960_720.jpg",
                ]

@st.cache  # Cache this so that it doesn't change every time something changes in the page
def load_default_dataset():
    return [load_image_from_url(url) for url in IMAGES_LINKS]

def load_image_from_url(url: str) -> Image.Image:
    return Image.open(requests.get(url, stream=True).raw)

@st.cache
def load_model() -> ClipModel:
    return ClipModel()


def limit_number_images(state: SessionState):
    """When moving between tasks sometimes the state of images can have too many samples"""
    if state.images is not None and len(state.images) > 1:
        state.images = [state.images[0]]


def limit_number_prompts(state: SessionState):
    """When moving between tasks sometimes the state of prompts can have too many samples"""
    if state.prompts is not None and len(state.prompts) > 1:
        state.prompts = [state.prompts[0]]


def is_valid_prediction_state(state: SessionState) -> bool:
    if state.images is None or len(state.images) < 1:
        st.error("Choose at least one image before predicting")
        return False
    if state.prompts is None or len(state.prompts) < 1:
        st.error("Write at least one prompt before predicting")
        return False
    return True


def preprocess_image(image: Image.Image, max_size: int = 1200) -> Image.Image:
    """Set up a max size because otherwise the API sometimes breaks"""
    width_0, height_0 = image.size

    if max((width_0, height_0)) <= max_size:
        return image

    if width_0 > height_0:
        aspect_ratio = max_size / float(width_0)
        new_height = int(float(height_0) * float(aspect_ratio))
        image = image.resize((max_size, new_height), Image.ANTIALIAS)
        return image
    else:
        aspect_ratio = max_size / float(height_0)
        new_width = int(float(width_0) * float(aspect_ratio))
        image = image.resize((max_size, new_width), Image.ANTIALIAS)
        return image


class Sections:
    @staticmethod
    def header():
        st.markdown('<link rel="stylesheet" '
                    'href="https://fonts.googleapis.com/css?family=Merriweather+Sans">'
                    '<style> '
                    'h1 {font-family: "Merriweather Sans", sans-serif; font-size: 48px; color: #f57c70}'
                    'a {color: #e6746a !important}'
                    '.stButton>button {'
                    '   color: white;'
                    '   background: #e6746a;'
                    '   display:inline-block;'
                    '   width: 100%;'
                    '   border-width: 0px;'
                    '   font-weight: 500;'
                    '   padding-top: 10px;'
                    '   padding-bottom: 10px;'
                    '}'
                    '</style>', unsafe_allow_html=True)
        st.markdown("# CLIP Playground")
        st.markdown("### Try OpenAI's CLIP model in your browser")
        st.markdown(" ")
        st.markdown(" ")
        with st.beta_expander("What is CLIP?"):
            st.markdown("CLIP is a machine learning model that computes similarity between text "
                        "(also called prompts) and images. It has been trained on a dataset with millions of diverse"
                        " image-prompt pairs, which allows it to generalize to unseen examples."
                        " <br /> Check out [OpenAI's blogpost](https://openai.com/blog/clip/) for more details",
                        unsafe_allow_html=True)
            col1, col2 = st.beta_columns(2)
            col1.image("https://openaiassets.blob.core.windows.net/$web/clip/draft/20210104b/overview-a.svg")
            col2.image("https://openaiassets.blob.core.windows.net/$web/clip/draft/20210104b/overview-b.svg")
        with st.beta_expander("What can CLIP do?"):
            st.markdown("#### Prompt ranking")
            st.markdown("Given different prompts and an image CLIP will rank the different prompts based on how well they describe the image")
            st.markdown("#### Image ranking")
            st.markdown("Given different images and a prompt CLIP will rank the different images based on how well they fit the description")
            st.markdown("#### Image classification")
            st.markdown("Similar to prompt ranking, given a set of classes CLIP can classify an image between them. "
                        "Think of [Hotdog/ Not hotdog](https://www.youtube.com/watch?v=pqTntG1RXSY&ab_channel=tvpromos) without any training.")
        st.markdown(" ")
        st.markdown(" ")

    @staticmethod
    def image_uploader(state: SessionState, accept_multiple_files: bool):
        uploaded_images = st.file_uploader("Upload image", type=[".png", ".jpg", ".jpeg"],
                                           accept_multiple_files=accept_multiple_files)
        if (not accept_multiple_files and uploaded_images is not None) or (accept_multiple_files and len(uploaded_images) >= 1):
            images = []
            if not accept_multiple_files:
                uploaded_images = [uploaded_images]
            for uploaded_image in uploaded_images:
                pil_image = Image.open(uploaded_image)
                pil_image = preprocess_image(pil_image)
                images.append(pil_image)
            state.images = images


    @staticmethod
    def image_picker(state: SessionState, default_text_input: str):
        col1, col2, col3 = st.beta_columns(3)
        with col1:
            default_image_1 = load_image_from_url("https://cdn.pixabay.com/photo/2014/10/13/21/34/clipper-487503_960_720.jpg")
            st.image(default_image_1, use_column_width=True)
            if st.button("Select image 1"):
                state.images = [default_image_1]
                state.default_text_input = default_text_input
        with col2:
            default_image_2 = load_image_from_url("https://cdn.pixabay.com/photo/2019/11/11/14/30/zebra-4618513_960_720.jpg")
            st.image(default_image_2, use_column_width=True)
            if st.button("Select image 2"):
                state.images = [default_image_2]
                state.default_text_input = default_text_input
        with col3:
            default_image_3 = load_image_from_url("https://cdn.pixabay.com/photo/2016/11/15/16/24/banana-1826760_960_720.jpg")
            st.image(default_image_3, use_column_width=True)
            if st.button("Select image 3"):
                state.images = [default_image_3]
                state.default_text_input = default_text_input

    @staticmethod
    def dataset_picker(state: SessionState):
        columns = st.beta_columns(5)
        state.dataset = load_default_dataset()
        image_idx = 0
        for col in columns:
            col.image(state.dataset[image_idx])
            image_idx += 1
            col.image(state.dataset[image_idx])
            image_idx += 1
        if st.button("Select random dataset"):
            state.images = state.dataset
            state.default_text_input = "A sign that says 'SLOW DOWN'"

    @staticmethod
    def prompts_input(state: SessionState, input_label: str, prompt_prefix: str = ''):
        raw_text_input = st.text_input(input_label,
                                    value=state.default_text_input if state.default_text_input is not None else "")
        state.is_default_text_input = raw_text_input == state.default_text_input
        if raw_text_input:
            state.prompts = [prompt_prefix + class_name for class_name in raw_text_input.split(";") if len(class_name) > 1]

    @staticmethod
    def single_image_input_preview(state: SessionState):
        st.markdown("### Preview")
        col1, col2 = st.beta_columns([1, 2])
        with col1:
            st.markdown("Image to classify")
            if state.images is not None:
                st.image(state.images[0], use_column_width=True)
            else:
                st.warning("Select an image")

        with col2:
            st.markdown("Labels to choose from")
            if state.prompts is not None:
                for prompt in state.prompts:
                    st.markdown(f"* {prompt}")
                if len(state.prompts) < 2:
                    st.warning("At least two prompts/classes are needed")
            else:
                st.warning("Enter the prompts/classes to classify from")

    @staticmethod
    def multiple_images_input_preview(state: SessionState):
        st.markdown("### Preview")
        st.markdown("Images to classify")
        col1, col2, col3 = st.beta_columns(3)
        if state.images is not None:
            for idx, image in enumerate(state.images):
                if idx < len(state.images) / 2:
                    col1.image(state.images[idx], use_column_width=True)
                else:
                    col2.image(state.images[idx], use_column_width=True)
            if len(state.images) < 2:
                col2.warning("At least 2 images required")
        else:
            col1.warning("Select an image")

        with col3:
            st.markdown("Query prompt")
            if state.prompts is not None:
                for prompt in state.prompts:
                    st.write(prompt)
            else:
                st.warning("Enter the prompt to classify")

    @staticmethod
    def classification_output(state: SessionState, model: ClipModel):
        # Possible way of customize this https://discuss.streamlit.io/t/st-button-in-a-custom-layout/2187/2
        if st.button("Predict") and is_valid_prediction_state(state):  # PREDICT 🚀
            with st.spinner("Predicting..."):
                
                st.markdown("### Results")
                # st.write(clip_response)
                if len(state.images) == 1:
                    scores = model.compute_prompts_probabilities(state.images[0], state.prompts)
                    scored_prompts = [(prompt, score) for prompt, score in zip(state.prompts, scores)]
                    st.json(scores)
                    sorted_scored_prompts = sorted(scored_prompts, key=lambda x: x[1], reverse=True)
                    for prompt, probability in sorted_scored_prompts:
                        percentage_prob = int(probability * 100)
                        st.markdown(
                            f"### ![prob](https://progress-bar.dev/{percentage_prob}/?width=200) &nbsp &nbsp {prompt}")
                elif len(state.prompts) == 1:
                    st.markdown(f"### {state.prompts[0]}")
                    
                    scores = model.compute_prompts_probabilities(state.images[0], state.prompts)
                    scored_images = [(image, score) for image, score in zip(state.images, scores)]
                    st.json(scores)
                    sorted_scored_images = sorted(scored_prompts, key=lambda x: x[1], reverse=True)

                    for image, probability in sorted_scored_images[:5]:
                        col1, col2 = st.beta_columns([1, 3])
                        col1.image(image, use_column_width=True)
                        percentage_prob = int(probability * 100)
                        col2.markdown(f"### ![prob](https://progress-bar.dev/{percentage_prob}/?width=200)")
                else:
                    raise ValueError("Invalid state")
                
                # is_default_image = isinstance(state.images[0], str)
                # is_default_prediction = is_default_image and state.is_default_text_input
                # if is_default_prediction:
                #     st.markdown("<br>:information_source: Try writing your own prompts and using your own pictures!",
                #                 unsafe_allow_html=True)
                # elif is_default_image:
                #     st.markdown("<br>:information_source: You can also use your own pictures!",
                #                 unsafe_allow_html=True)
                # elif state.is_default_text_input:
                #     st.markdown("<br>:information_source: Try writing your own prompts!"
                #                 " It can be whatever you can think of",
                #                 unsafe_allow_html=True)


Sections.header()
col1, col2 = st.beta_columns([1, 2])
col1.markdown(" "); col1.markdown(" ")
col1.markdown("#### Task selection")
task_name: str = col2.selectbox("", options=["Prompt ranking", "Image ranking", "Image classification"])
st.markdown("<br>", unsafe_allow_html=True)

model = load_model()
session_state = get_state()
if task_name == "Image classification":
    Sections.image_uploader(session_state, accept_multiple_files=False)
    if session_state.images is None:
        st.markdown("or choose one from")
        Sections.image_picker(session_state, default_text_input="banana; boat; bird")
    input_label = "Enter the classes to chose from separated by a semi-colon. (f.x. `banana; boat; honesty; apple`)"
    Sections.prompts_input(session_state, input_label, prompt_prefix='A picture of a ')
    limit_number_images(session_state)
    Sections.single_image_input_preview(session_state)
    Sections.classification_output(session_state, model)
elif task_name == "Prompt ranking":
    Sections.image_uploader(session_state, accept_multiple_files=False)
    if session_state.images is None:
        st.markdown("or choose one from")
        Sections.image_picker(session_state, default_text_input="A calm afternoon in the Mediterranean; "
                                                                "A beautiful creature;"
                                                                " Something that grows in tropical regions")
    input_label = "Enter the prompts to choose from separated by a semi-colon. " \
                  "(f.x. `An image that inspires; A feeling of loneliness; joyful and young; apple`)"
    Sections.prompts_input(session_state, input_label)
    limit_number_images(session_state)
    Sections.single_image_input_preview(session_state)
    Sections.classification_output(session_state, model)
elif task_name == "Image ranking":
    Sections.image_uploader(session_state, accept_multiple_files=True)
    if session_state.images is None or len(session_state.images) < 2:
        st.markdown("or use this random dataset")
        Sections.dataset_picker(session_state)
    Sections.prompts_input(session_state, "Enter the prompt to query the images by")
    limit_number_prompts(session_state)
    Sections.multiple_images_input_preview(session_state)
    Sections.classification_output(session_state, model)

st.markdown("<br><br><br><br>Made by [@JavierFnts](https://twitter.com/JavierFnts) | [How was CLIP Playground built?](https://twitter.com/JavierFnts/status/1363522529072214019)"
            "", unsafe_allow_html=True)
session_state.sync()
