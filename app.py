import cv2 # untuk import library opencv
import streamlit as st # untuk import library streamlit
import numpy as np # untuk import library numpy

from PIL import Image # untuk import library Pillow

import tensorflow as tf # untuk import library Tensorflow
from keras import backend as K # untuk import library Keras
from streamlit_drawable_canvas import st_canvas # untuk import library streamlit_drawable_canvas

def preprocess(img): # fungsi untuk preproses gambar sesuai ukuran 256 x 64 pixel
    (h, w) = img.shape # mengambil ukuran panjang dan lebar gambar
    
    fix_img = np.ones([64, 256])*255 # membuat array ukuran 64 x 256 dengan isi 255 yaitu warna putih
    
    if w > 256: # cek ukuran lebar jika lebih dari 256 pixel
        img = img[:, :256] # jika lebih maka diganti ke 256 pixel
        
    if h > 64: # cek ukuran tinggi jika lebih dari 64 pixel
        img = img[:64, :] # jika lebih maka diganti ke 64 pixel
    
    fix_img[:h, :w] = img # menerapkan ukuran tinggi dan lebar di fix image
    return cv2.rotate(fix_img, cv2.ROTATE_90_CLOCKWISE) # output hasil gambar yg telah diganti ukurannya dan di rotate

abjad = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' " # membuat variabel utk abjad dan karakter

def angkalabel(num): # konversi angka ke label karna hasil prediksi merupakan urutan abjad 
    ret = "" # define variabel kosong
    for ch in num: # loop berdasarkan input parameter
        if ch == -1: # jika hasil -1 maka berhenti
            break
        else: 
            ret+=abjad[ch] # jika terdapat hasil abjad, maka variabel kosong diisi dengan hasil
    return ret # output hasil konversi


def main(): # fungsi utama untuk menjalankan streamlit
    PAGES = { # dictionary
        "Upload Image": up_img, # utk halaman upload gambar
        "Canvas": canvas_app # utk halaman canvas
    }
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys())) # memasukkan dictionary ke dalam dropdown
    PAGES[page]()

    with st.sidebar: # utk tampilan sidebar penjelasan
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://github.com/kurniawan2805/handwriting-recognition">@Tim HR 4</a> </h6>',
            unsafe_allow_html=True,
        )


def up_img(): # fungsi utk prediksi berdasarkan upload gambar

    model = tf.keras.models.load_model('temp_model/hrv1.h5') # load model yang sudah di latih

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg']) # streamlit utk upload gambar dengan tipe yang telah ditentukan
    if not image_file: # jika tidak gambar upload, maka output tidak ada
        return None

    original_image = Image.open(image_file) # utk open gambar menggunakan pillow
    original_image = np.array(original_image) # utk konversi gambar ke dalam bentuk array

    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) # konversi gambar ke warna abu abu

    image = preprocess(image) # menjalankan fungsi prepocess terhadap gambar
    image = image/255. # membagi nilai array gambar dengan 255 sehingga rentang nilainya dari 0 ke 1
    
    pred = model.predict(image.reshape(1, 256, 64, 1)) # prediksi gambar menggunakan model
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                       greedy=True)[0][0]) # hasil prediksi dalam bentuk angka

    st.text(f"Image Handwriting") # menampilkan teks
    st.image([image_file]) # menampilkan gambar
    st.text(f"Probably the result: {angkalabel(decoded[0])}") # menampilkan teks dan hasil yang telah dikonversi ke tulisan

def canvas_app(): # fungsi untuk input tulisan melalui canvas (masih pengembangan)
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3) # ketebalan garis tulisan di canvas
    stroke_color = st.sidebar.color_picker("Stroke color hex: ") # warna garis
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee") # warna background

    hasil_tulis = st_canvas( # canvas komponen
        fill_color="rgba(255, 165, 0, 0.3)",  # mengisi warna 
        stroke_width=stroke_width, # atur ketebalan garis
        stroke_color=stroke_color, # atur warna garis
        background_color=bg_color, # atur background
        height=64, # atur tinggi
        width=256, # atur lebar
        drawing_mode="freedraw", # freedraw untuk mode gambar bebas seperti pena
        key="kanvas", # nama kanvas
    )

    if hasil_tulis.image_data is not None: # cek jika ada tulisan di kanvas
        model = tf.keras.models.load_model('temp_model/hrv1.h5') # load model yang telah dilatih

        imagefile = hasil_tulis.image_data # data gambar dari kanvas

        image = imagefile.copy() # duplikat data gambar
        image = image.astype('uint8') # konversi bentuk array 0 - 255

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # konversi warna ke abu abu

        image = preprocess(image) # jalankan fungsi preprocess
        image = image/255. # array jadi 0 - 1
        
        pred = model.predict(image.reshape(1, 256, 64, 1)) # gambar diprediksi menggunakan model
        decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                        greedy=True)[0][0]) # hasil berupa angka

        st.text("Image Handwriting") # tampilkan tulisan
        st.image([imagefile]) # tampilkan gambar
        st.text(f"Probably the result : {angkalabel(decoded[0])}") # tampilkan tulisan dan hasil yang telah dikonversikan ke tulisan

if __name__ == '__main__': # fungsi yang dijalankan diawal
    st.set_page_config( # mengatur judul web dan ikon
        page_title="Streamlit Handwriting Recognition Demo", page_icon="🖊️" 
    )
    st.title("Handwriting Recognition Demo App") # mengatur judul halaman
    st.subheader("This app allows you to recognize handwriting image !") # mengatur judul sub halaman
    st.text("We use Tensorflow and Streamlit for this demo") # menampilkan tulisan
    st.sidebar.subheader("Configuration") # tulisan di sidbar
    main() # menjalankan fungsi main