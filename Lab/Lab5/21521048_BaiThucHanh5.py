import streamlit as st
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tensorflow.keras.preprocessing import image
import joblib

IMG_SIZE = 227

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0  # Chuẩn hóa ảnh
    img = np.expand_dims(img, axis=0)  # Thêm chiều batch
    return img

def predict_image(model, img_path):
    img = preprocess_image(img_path)
    img_flat = img.reshape(-1, IMG_SIZE * IMG_SIZE * 3)  # Chuyển ảnh thành vector 1D
    if isinstance(model, LogisticRegression):
        prediction = model.predict(img_flat)
    elif isinstance(model, SVC):
        prediction = model.predict(img_flat)
    return prediction

def load_model():
    model = joblib.load('logreg_model.pkl')
    return model

def main():
    st.title("Ứng dụng Dự Đoán Viêm Phổi từ X-Quang")
    st.write("Tải lên tấm ảnh X-quang để dự đoán có viêm phổi hay không")

    uploaded_file = st.file_uploader("Chọn ảnh X-quang", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Hiển thị ảnh lên giao diện
        st.image(uploaded_file, caption='Ảnh X-quang', use_column_width=True)

        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        model = load_model()

        prediction = predict_image(model, "temp_image.jpg")
        result = "PNEUMONIA" if prediction == 1 else "NORMAL"

        st.write(f"Dự đoán: {result}")

if __name__ == "__main__":
    main()
