import streamlit as st
import cv2
import numpy as np


# convert image to pencil sketch
def pencil_sketch(img):
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    # inverted_gray = 255 - gray_img
    inverted_gray = cv2.bitwise_not(gray_img)

    # Blur the inverted image
    blurred_img = cv2.GaussianBlur(inverted_gray, (21, 21), 0)

    inverted_blur = cv2.bitwise_not(blurred_img)

    # Mix image
    pencil_sketch_img = cv2.divide(gray_img, inverted_blur, scale=256)

    return pencil_sketch_img


def save_image(sketch, file_name):
    # Ensure the file name has a valid extension

    valid_extensions = ['.jpg', '.jpeg', '.png']
    if not any(file_name.endswith(ext) for ext in valid_extensions):
        file_name += '.png'  # Default to PNG if no valid extension provided

    try:
        # Debug information
        st.write(f"Attempting to save file: {file_name}")
        st.write(f"Image shape: {sketch.shape}")
        st.write(f"Image dtype: {sketch.dtype}")

        success = cv2.imwrite(file_name, sketch)
        if success:
            st.success(f"File saved successfully as {file_name}")
        else:
            st.error("Failed to save the image. Please check the file name and path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")


# st.title("Image to Pencil Sketch App")
st.header("Image to Pencil Sketch App")

st.write("Upload an image and convert it to a pencil sketch!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    st.image(img, caption='Original Image', use_column_width=True)
    sketch = pencil_sketch(img)
    st.image(sketch, caption='Pencil Sketch', use_column_width=True)
    file_name = st.text_input("Enter File Name")
    if st.button("Save Image"):
        if file_name is not None:
            save_image(sketch,file_name)




