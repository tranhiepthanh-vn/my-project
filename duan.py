# khai báo thư viện và api_key
import time
from datetime import datetime, timedelta
from plyer import notification
import os
import pandas as pd

os.environ["GOOGLE_API_KEY"] = ""

# text to speech

def noi(x):
    from gtts import gTTS
    from io import BytesIO
    import pygame
    tts = gTTS(text=x, lang="vi")
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)

    # Phát âm thanh

    mp3_fp.seek(0)
    pygame.mixer.init()
    pygame.mixer.music.load(mp3_fp, "mp3")
    pygame.mixer.music.play()

    # Chờ cho đến khi phát xong

    while pygame.mixer.music.get_busy():
        time.sleep(0.5)
    pygame.mixer.quit()

# nhận diện giọng nói(speech to text)

def nhangiong():
    import pyaudio
    import speech_recognition as sr
    import asyncio
    import base64
    import json
    s = sr.Recognizer()
    with sr.Microphone() as source:
        print("Đang lắng nghe")
        mp3 = s.listen(source)
    vb = s.recognize_google(mp3, language="vi-VN")
    try:
        vb
    except:
        print("Không có âm thanh")
    return vb.lower()
#quay lại kiểm tra
def molai():
    txt='xin chào'
    noi("xin chào")
    # kiểm tra yêu cầu
    while True:
        if "hello" or "xin chào" in txt:
                
            noi("xin hỏi yêu cầu của bạn là gì")
                
            # nói yêu cầu
                
            xt =input("yêu cầu:")
            if 'xóa' in xt:
                hienthi()
                xoa()
            elif 'tạm biệt' in xt:
                noi("Chào tạm biệt bạn")
                break
            elif 'hỏi đáp' in xt:
                noi("Đang khởi động hỏi đáp")
            elif 'xem' in xt:
                noi("Đang in lịch trình")
                hienthi()
            elif 'gemini' in xt:
                gemini()
            else:
                noi("Xin lỗi,tôi không có chức năng này")
# xét file lịch trình
file = "D:/CODER/python/xlsx/lichtrinh.xlsx"

lan_nhac_gan_nhat = {}
# nếu chưa có,tạo file mới
if not os.path.exists(file):
    noi("Hiện chưa có lịch trình, tôi sẽ tạo file mới.")
    df = pd.DataFrame(columns=["tiêu đề", "thời gian"])
    df.to_excel(file, index=False)
    noi("Đã tạo xong")
# xem có đang nằm hay không
import cv2
from ultralytics import YOLO

model = YOLO("yolo11l.pt")

def dang_nam_hien_tai():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return False

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        if model.names[int(box.cls[0])] == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            if h > 0 and w / h > 1.4:
                return True

    return False

# ===== THÔNG BÁO =====
def thong_bao(tieu_de, noi_dung):
    notification.notify(
        title=tieu_de,
        message=noi_dung,
        timeout=10
    )
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {tieu_de}: {noi_dung}")

# ===== KIỂM TRA & NHẮC =====
def kiem_tra_lich():
    df = pd.read_excel(file)

    # Chuẩn hoá cột thời gian (AUTO nhận dạng mọi định dạng Excel)
    df["thoi_gian"] = pd.to_datetime(
        df["thời gian"],
        errors="coerce",
    )

    now = datetime.now()

    for _, row in df.iterrows():
        if pd.isna(row["thoi_gian"]):
            continue

        thoi_diem_goc = row["thoi_gian"]
        bat_dau_nhac = thoi_diem_goc - timedelta(minutes=30)

        # Chỉ nhắc trong khoảng cho phép
        if not (bat_dau_nhac <= now < thoi_diem_goc):
            continue

        key = f"{thoi_diem_goc}_{row['tiêu đề']}"
        lan_nhac_cuoi = lan_nhac_gan_nhat.get(key)

        if lan_nhac_cuoi is None or now - lan_nhac_cuoi >= timedelta(minutes=5):
            thong_bao(
                f"⏰ SẮP ĐẾN GIỜ ({thoi_diem_goc.strftime('%H:%M')})",
                f"{row['tiêu đề']}")
            noi("gần đến giờ thực hiện lịch hôm nay rồi")
            if dang_nam_hien_tai():
                thong_bao(
                    f"🛏️ SẮP ĐẾN GIỜ ({thoi_diem_goc.strftime('%H:%M')})",
                    f"{row['tiêu đề']} (bạn đang nằm)"
                )
                noi("hãy đứng dạy nào, sắp đến giờ thực hiện lịch rồi nhé")

#vòng lặp nhắc

def vong_lap_nhac():
    while True:
        try:
            kiem_tra_lich()
        except Exception as e:
            print("❌ Lỗi kiểm tra lịch:", e)

        time.sleep(120)  # kiểm tra mỗi 2p
#chạy nền

def chay_thread_nhac():
    import threading
    thread = threading.Thread(
        target=vong_lap_nhac,
    )
    thread.start()
    return thread

# xoá lịch
def xoa():
    noi("Hãy nói tên lịch cần xoá")
    noidung = nhangiong()

    df = pd.read_excel(file)

    # kiểm tra có tồn tại không
    ndxoa = df["tiêu đề"].str.lower() == noidung
    if ndxoa.any():
        df = df[~ndxoa]
        df.to_excel(file, index=False)

        noi("Đã xoá lịch trình")
    else:
        noi("Không tìm thấy lịch cần xóa")
# hiển thị
def hienthi():
    df=pd.read_excel("lichtrinh.xlsx")
    print(df)

# Nhận dạng cảm xúc thông qua hình ảnh và giọng nói
def emotion():
    import joblib

    model = joblib.load("D:/CODER/python/emotion_model/voice_emotion_model.pkl")
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    import librosa

    def extract_features(file_path, duration=4):
        y, sr = librosa.load(file_path, sr=22050, duration=duration)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)

        zcr = librosa.feature.zero_crossing_rate(y).mean()
        rms = librosa.feature.rms(y=y).mean()

        return np.hstack([mfcc_mean, zcr, rms])

    def du_doan_cam_xuc_tu_file(wav_path):
        features = extract_features(wav_path)
        features = np.array(features).reshape(1, -1)

        label = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        if label == 1:
            return "TÍCH CỰC", prob[1]
        else:
            return "TIÊU CỰC", prob[0]

    def nhan_dien_cam_xuc_tu_micro(duration=4, fs=22050):
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()

        sf.write("temp.wav", audio, fs)

        return du_doan_cam_xuc_tu_file("temp.wav")
    emotion, confidence = nhan_dien_cam_xuc_tu_micro()
    if emotion=="TIÊU CỰC" and confidence>=0.5:
        noi("xin chào,bạn có ổn không?")
    elif emotion=="TIÊU CỰC" and confidence<=0.5:
        noi("ngày hôm nay của bạn như thế nào?")
    chaycode()

# kết nối với gemini thông qua langgraph
def gemini():
    from dotenv import load_dotenv
    from typing_extensions import TypedDict

    import google.generativeai as genai
    from langgraph.graph import StateGraph, START, END

    # ===== LOAD API KEY =====
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # ===== INIT GEMINI MODEL =====
    model = genai.GenerativeModel("gemini-2.5-flash")

    # ===== STATE (giống Colab) =====
    class ChatState(TypedDict):
        messages: list[str]

    # ===== HUMAN NODE =====
    def human_node(state: ChatState) -> ChatState:
        user_input = input("USER:")

        if user_input.lower() in ["tạm dừng", "thoát"]:
            print("đang tắt gemini")
            noi("đang tắt gemini")
            raise KeyboardInterrupt  # thoát graph an toàn

        return {
            "messages": state["messages"] + [user_input]
        }

    # ===== CHATBOT NODE =====
    def chatbot_node(state: ChatState) -> ChatState:
        # lấy 1-2 tin nhắn gần nhất để làm context
        user_input = state["messages"][-1]
        context = ("Trả lời chi tiết vừa đủ, không markdown.\n"+ user_input)

        response = model.generate_content(context,generation_config={
            "max_output_tokens": 256,
            "temperature": 0.7
        })
        reply = response.text.strip()

        print(f"\n🤖 Chatbot: {reply}\n")
        noi(f"{reply}\n")
        return {
            "messages": state["messages"] + [reply]
        }

    # xây dựng langgraph và các node
    graph = StateGraph(ChatState)

    graph.add_node("human", human_node)
    graph.add_node("chatbot", chatbot_node)

    graph.add_edge(START, "human")
    graph.add_edge("human", "chatbot")
    graph.add_edge("chatbot", END)

    app = graph.compile()

    # khi khởi động xong
    if __name__ == "__main__":
        print(" Chatbot Gemini sẵn sàng\n")
        noi(" Chatbot Gemini sẵn sàng\n")

        state: ChatState = {"messages": []}

        while True:
            try:
                state = app.invoke(state)
            except KeyboardInterrupt:
                break
chay_thread_nhac()
def chaycode():
    txt='xin chào'
    # kiểm tra yêu cầu
    while True:
        if "hello" or "xin chào" in txt:
                
            noi("chào bạn,xin hỏi yêu cầu của bạn là gì")
                
            # nói yêu cầu
                
            xt =input("yêu cầu:")
            if 'xóa' in xt:
                hienthi()
                xoa()
            elif 'tạm biệt' in xt:
                noi("Chào tạm biệt bạn")
                break
            elif 'hỏi đáp' in xt:
                noi("Đang khởi động hỏi đáp")
            elif 'xem' in xt:
                noi("Đang in lịch trình")
                hienthi()
            elif 'gemini' in xt:
                gemini()
            else:
                noi("Xin lỗi,tôi không có chức năng này")           
emotion()
input("nhấn enter để mở lại chatbot")

molai()
