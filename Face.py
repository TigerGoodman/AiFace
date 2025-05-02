import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("ตรวจจับใบหน้าในภาพ")

uploaded_file = st.file_uploader("อัปโหลดภาพจากมือถือ", type=["jpg","png","webp"])

if uploaded_file is not None:
    #อ่านภาพด้วย PIL แล้วแปลงเป็น OpenCV
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # โหลดตัวตรวจจับใบหน้า (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # แปลงภาพเป็น grayscale เพื่อใช้ตรวจจับใบหน้า
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor = 1.1,
    minNeighbors = 9,
    minSize = (200,200))

    # วาดกรอบรอบใบหน้า
    for (x,y,w,h) in faces:
        cv2.rectangle(img_np,(x,y),(x + w, y + h),(255, 0, 0), 2)

    # แสดงผล
    st.image(img_np, caption = f"ตรวจพบใบหน้า {len(faces)} คน", use_container_width = True)
    st.success(f"ตรวจพบใบหน้าทั้งหมด: {len(faces)}คน")
