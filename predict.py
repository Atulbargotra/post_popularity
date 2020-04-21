import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import time
addr = 'http://localhost:5000'
test_url = addr + '/api/predict'
content_type = 'image/jpeg'
headers = {'content-type': content_type}
st.title("Predict Your Instagram Post Rating 🤟")
file_bytes = st.file_uploader("Upload a file", type=("png", "jpg","jpeg"))
if file_bytes is not None:
    image = Image.open(file_bytes)
    #st.image(image,use_column_width=True)
    image_arr = np.array(image)
    cv2_image = image_arr[:, :, ::-1].copy()
    cv2_image = cv2.resize(cv2_image,(224,224))
    _, img_encoded = cv2.imencode('.jpg', cv2_image)
    response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
    #st.subheader(response.text)
    with st.spinner('Doing Math...'):
        time.sleep(2)
        st.success('Done!')
    st.subheader(f'Prediction {response.text}')
    st.balloons()


