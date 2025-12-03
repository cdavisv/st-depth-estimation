"""
Streamlit app for depth estimation using DepthAnything model.
Select images with objects you know the distance to, click a point on the image, and input the distance. 
This will connect into the AddaxAI streamlit application. 
"""
# TODO: Implement DepthAnything model inference and visualization.

import os
import streamlit as st
from streamlit_image_select import image_select
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
from PIL import Image


st.set_page_config(layout="wide")
st.title("Depth Estimation Tool")

# Hide the text cursor in the selectbox input
st.markdown(
    """
    <style>
    /* Hide the blinking cursor in Streamlit's selectbox search input */
    div[data-baseweb="select"] input {
        caret-color: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "test-imgs")
CSV_PATH = os.path.join(IMAGE_FOLDER, "results_detections.csv")

# predictions
if os.path.exists(CSV_PATH):
    predictions = pd.read_csv(CSV_PATH)
    st.write("### Predictions CSV")
    st.dataframe(predictions)
else:
    st.error(f"CSV not found at: {CSV_PATH}")
    st.stop()

# images
images = [
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

if not images:
    st.error(f"No images found in folder: {IMAGE_FOLDER}")
    st.stop()

st.write("### Image Selector")
# Select number of images to display
num_images = st.selectbox(
    "How many images would you like to display?",
    options=[4, 8, 12, 16, 20, len(images)],
    index=0  # default = first option
)

# Limit the images passed to image_select
limited_images = images[:num_images]

# Display image selector
selected_image = image_select(
    "Select an image to input distance measurement.",
    limited_images,
    use_container_width=True
)

# --------------------------------------------------------------------------------
# IMAGE POINT SELECTION + DISTANCE ENTRY
# --------------------------------------------------------------------------------
if selected_image:

    st.write("### Selected Image:")
    filename = os.path.basename(selected_image)
    st.write(filename)

    pil_img = Image.open(selected_image)

    st.write("### Click a point on the image")
    st.info("Click the object whose distance you know (example: a tree trunk).")

    # Get click coordinates on the image
    click_result = streamlit_image_coordinates(
        pil_img,
        key="click_coords"
    )

    clicked_point = None
    if click_result is not None:
        # streamlit-image-coordinates returns a dict with x, y, and timestamp
        clicked_point = (int(click_result["x"]), int(click_result["y"]))
        st.success(f"Clicked pixel: {clicked_point}")

    st.write("### Enter known distance to this point")
    distance_m = st.text_input("Distance (meters)", key="distance_input")

    # -------------------------------------------------------------------
    # SAVE DISTANCE + CLICKED POINT
    # -------------------------------------------------------------------
    if st.button("Save Distance & Point"):
        if clicked_point is None:
            st.error("Please click a point on the image first.")
            st.stop()

        if distance_m.strip() == "":
            st.error("Please enter a distance.")
            st.stop()

        # Match on filename within relative_path column
        if "relative_path" not in predictions.columns:
            st.error("CSV must contain a 'relative_path' column.")
            st.stop()

        mask = predictions["relative_path"].str.contains(filename)

        if not mask.any():
            st.error(f"Could not find {filename} in CSV relative_path column!")
            st.stop()

        # Add missing columns if needed
        for col in ["distance_m", "px_x", "px_y"]:
            if col not in predictions.columns:
                predictions[col] = ""

        # Save values
        predictions.loc[mask, "distance_m"] = distance_m
        predictions.loc[mask, "px_x"] = clicked_point[0]
        predictions.loc[mask, "px_y"] = clicked_point[1]

        predictions.to_csv(CSV_PATH, index=False)

        st.success(
            f"Saved for {filename}: distance={distance_m}m, point={clicked_point}"
        )
        st.info("These values will be used later for DepthAnything V3 calibration.")