import os
import streamlit as st
from pathlib import Path
from PIL import Image

from MewUtilities.mew_streamlit import *

mew_header_st()

def load_images(directory):
    image_files = [f for f in directory.rglob('*') if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}]

    images = []
    for image_file in image_files:
        images.append(Image.open(image_file))
    return images

def display_images(images):
    def display_image(image):
        with st.container(border=True):
            st.image(image, image, use_column_width=True)

    # Streamlit app
    #st.header('Dream Images')

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

def main():
    st.title('Image Explorer')

    # Get the root directory
    root_directory = Path('data/image')

    # Get all subdirectories
    subdirectories = [d for d in root_directory.iterdir() if d.is_dir()]

    # Create tabs for each subdirectory
    subdir = st.radio("Select a subdirectory:", subdirectories)
    with subdir:
        with st.container(border=True):
            st.title(f"Images in {subdir.name}")
            images = load_images(subdir)
            if not images:
                st.warning(f'No images found in {subdir}.')
            else:
                display_images(images)

if __name__ == '__main__':
    main()
