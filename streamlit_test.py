# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 18:53:15 2022

@author: kurni
"""
# from matplotlib import container
import numpy as np
from numpy import blackman
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# import matplotlib.pyt
import os
import cv2
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model


st.markdown(
    """
<style>
.reportview-container .markdown-text-container {
    font-family: monospace;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
.Widget>label {
    color: white;
    font-family: monospace;
}
[class^="st-b"]  {
    color: white;
    font-family: monospace;
}
.st-bb {
    background-color: transparent;
}
.st-at {
    background-color: #0c0080;
}
footer {
    font-family: monospace;
}
.reportview-container .main footer, .reportview-container .main footer a {
    color: #0c0080;
}
header .decoration {
    background-image: none;
}

</style>
""",
    unsafe_allow_html=True,
)


#model parameter
batch_size = 64
padding_token = 99
image_width = 256
image_height = 64
max_len = 32

model_path = "model_tuned.h5"
prediction_model = load_model(model_path, compile=False)

test=pd.read_csv('written_name_test_v2.csv')
test['IDENTITY']=test['IDENTITY'].str.upper()

characters = set()
characters = [' ', "'", '-', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] 
max_str_len = 16 # max length of input labels
num_of_characters = len(characters) + 1 # +1 for ctc pseudo blank

# from  tensorflow.keras.layers.experimental.preprocessing import StringLookup
# Convert characters to integers.
char_to_num = tf.keras.layers.StringLookup(vocabulary=list(characters), mask_token=None)
AUTOTUNE = tf.data.AUTOTUNE

# Convert integers back to original characters.
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize_with_pad(image, h, w)
    pad_width = w - tf.shape(image)[1]

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image(image_path, img_size=(image_width, image_height)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, 1)
    image = distortion_free_resize(img, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def load_image(image_file):
	img = Image.open(image_file)
	return img

def draw_canvas():
    with st.container():
    # with col1:
        st.write('Tulis nama Anda di Canvas dengan huruf kapital!')
        # canvas_result = create_canvas(realtime_update)
        scale = 3
        # Specify canvas parameters in application
        stroke_width = 2
        stroke_color = "Black"
        bg_color = "White"
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image= None, #Image.open(bg_image) if bg_image else None,
            width = 256*scale,
            height= 64*scale,
            # update_streamlit=realtime_update
            key="canvas",
        )
    # with col2:
    show_results = st.button('Predict Text!')
    if show_results:
        with st.container():
            if canvas_result is not None:
                try:
                    # # "Hello World" mm
                    image_re = canvas_result.image_data[:, :, 1]
                    # image_re = image_re.astype(float) // 255
                    im = Image.fromarray(image_re.astype('uint8'))
                    # save_img("test.png", image_data)
                    
                    im.save("test.jpg", "JPEG")

                    # process img
                    batch_images=np.ones((batch_size,256,64,1),dtype=np.float32)
                    image = preprocess_image("test.jpg")
                    
                    batch_images[0]=image
                    x=prediction_model.predict(batch_images)
                    pred_texts = decode_batch_predictions(x)

                    st.write('Drawn Image (Scaled)')
                    st.image("test.jpg", width=256)

                    pred_text = pred_texts[0]
                    st.write('Predicted Text:',pred_text)

                except Exception:
                    pass
def upload_image():
    image_file = st.file_uploader("Upload gambar tulisan", type=['jpg', 'png', 'jpeg']) # streamlit utk upload gambar dengan tipe yang telah ditentukan
    if not image_file: # jika tidak gambar upload, maka output tidak ada
        return None
    if image_file is not None:

        # To See details
        file_details = {"filename":image_file.name, "filetype":image_file.type,
                        "filesize":image_file.size}
        st.write(file_details)

        # To View Uploaded Image
        # st.image(load_image(image_file),width=256)

        with open(os.path.join("test3.jpg"),"wb") as f:
            f.write((image_file).getbuffer())

        batch_images=np.ones((batch_size,256,64,1),dtype=np.float32)
        image = preprocess_image("test3.jpg")
        batch_images[0]=image

        x=prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(x)
        pred_text = pred_texts[0]

        # st.text(f"Image Handwriting") # menampilkan teks
        with st.container():
            st.image(image_file, width=256) # menampilkan gambar
            st.text(f"Predicted text: {pred_text}") # menampilkan teks dan hasil yang telah dikonversi ke tulisan
            # check if label exist
            res = test.isin([image_file.name]).any().any()
            if res:
                truth = test.loc[test['FILENAME']==image_file.name ].IDENTITY.iat[0]
                st.text(f"Ground truth: {truth}")

def main():

    st.title('Handwriting Recognition')
    st.write('Aplikasi untuk mendeteksi teks gambar tulisan tangan')
 
    st.sidebar.title('Navigate')
    navigation_mode = st.sidebar.radio('', ['Upload Image', 'Draw Canvas'])

    if navigation_mode == 'Draw Canvas':
        draw_canvas()
    if navigation_mode == 'Upload Image':
        upload_image()        

if __name__ == '__main__':
    
    

    main()
