import os
import streamlit as st

from pathlib import Path
from PIL import Image

from MewUtilities.mew_streamlit import *

mew_header_st()

# Directory path where images are stored
image_directory = 'data/image/'

# Get a list of all image files in the directory
image_files = [f for f in Path(image_directory).rglob('*') if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}]

images = []
for image_file in image_files:
    images.append(Image.open(image_file))



def display_dream_images(images):
    def display_image(image):
        with st.container(border=True):
            st.image(image, image, use_column_width=True)

    # Streamlit app
    st.header('Dream Images')

    # Display images in a grid layout
    num_images = len(images)
    if num_images == 0:
        st.warning('No images found.')
    else:
        max_columns = 3
        num_columns = min(num_images, max_columns)  # Display up to 3 columns
        rows = [images[i:i + num_columns] for i in range(0, num_images, num_columns)]

        for row in rows:
            col1, col2, col3 = st.columns(num_columns)
            for i, image in enumerate(row):
                with locals()[f"col{i+1}"]:
                    display_image(image)

display_dream_images(images)