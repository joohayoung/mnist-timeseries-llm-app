import streamlit as st
import io
import os
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


import torch
import torch.nn as nn
import torchvision.transforms as transforms

from tensorflow.keras.models import load_model
import joblib
import datetime as dt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# =========================================
# GEMINI 설정 (공통)
# =========================================
API_KEY = st.secrets["GEMINI_API_KEY"]

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.5-flash-lite:generateContent?key={API_KEY}"
)

def ask_gemini(prompt):
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    res = requests.post(GEMINI_URL, headers=headers, json=data)
    try:
        return res.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "LLM 응답 파싱 실패"


# =========================================
# MNIST 페이지
# =========================================
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


@st.cache_resource
def load_mnist_model():
    model = MNISTModel()
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))
    model.eval()
    return model


mnist_model = load_mnist_model()

mnist_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def mnist_page():
    st.title("✍️ 손글씨 숫자 인식 (MNIST)")
    st.write("이미지를 업로드하면 MNIST 모델이 숫자를 예측하고 Gemini가 설명을 제공합니다.")

    uploaded = st.file_uploader("📤 이미지 업로드", type=["png", "jpg", "jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="업로드된 이미지", width=200)

        # 변환
        img_tensor = mnist_transform(image).unsqueeze(0)

        with torch.no_grad():
            output = mnist_model(img_tensor)
            digit = output.argmax(dim=1).item()

        st.subheader(f"예측된 숫자: **{digit}**")

        with st.spinner("Gemini가 부드러운 설명 생성 중..."):
            prompt = f"이 손글씨 이미지는 MNIST 모델 기준 숫자 {digit}로 분류되었어. 사용자에게 친절하게 설명해줘."
            explanation = ask_gemini(prompt)

        st.write(explanation)


# =========================================
# LSTM 페이지
# =========================================
@st.cache_resource
def load_lstm():
    model = load_model("lstm_model.h5")
    scaler = joblib.load("lstm_scaler.pkl")
    return model, scaler

lstm_model, lstm_scaler = load_lstm()


def predict_future(model, scaled_data, look_back=60, future_steps=200):
    seq = scaled_data[-look_back:].reshape(1, look_back, 1)
    preds = []
    for _ in range(future_steps):
        next_val = model.predict(seq, verbose=0)[0, 0]
        preds.append(next_val)
        seq = np.append(seq[:, 1:, :], [[[next_val]]], axis=1)
    return np.array(preds)


def lstm_page():
    st.title("📈 LSTM 기반 Time-Series 예측")
    st.write("CSV/Excel 파일을 업로드하면 **200개 값**을 예측합니다.")

    uploaded_file = st.file_uploader("📤 CSV 또는 Excel", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if "time" not in df.columns or "value" not in df.columns:
            st.error("time, value 컬럼이 필요합니다.")
            return

        df["time"] = pd.to_datetime(df["time"])
        st.dataframe(df.head())

        scaled = lstm_scaler.transform(df["value"].values.reshape(-1, 1)).reshape(-1)

        future_scaled = predict_future(lstm_model, scaled)
        future_pred = lstm_scaler.inverse_transform(future_scaled.reshape(-1, 1)).reshape(-1)

        last_date = df["time"].iloc[-1]
        future_dates = pd.date_range(last_date + dt.timedelta(days=1), periods=200)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["time"], df["value"], label="Actual", linewidth=2)
        ax.plot(future_dates, future_pred, label="Predicted", color="orange", linestyle="--")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)

        # LLM 분석
        st.subheader("Gemini 요약")
        prompt = f"""
        시계열 예측 결과 요약:
        - 실제 데이터: {len(df)}개
        - 예측: 200개
        - 마지막 실제 값: {df['value'].iloc[-1]}
        - 예측 첫 값: {future_pred[0]}
        - 예측 마지막 값: {future_pred[-1]}
        친절하고 부드럽게 요약해줘.
        """
        st.write(ask_gemini(prompt))


# =========================================
#  페이지 라우팅
# =========================================
st.sidebar.title("📌 메뉴 선택")
page = st.sidebar.radio("서비스를 선택하세요", ["MNIST 손글씨 인식", "LSTM 시계열 예측"])

if page == "MNIST 손글씨 인식":
    mnist_page()
else:
    lstm_page()
