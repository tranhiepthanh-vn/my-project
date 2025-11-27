# khai báo thư viện và api_key
import time
from datetime import datetime, timedelta
from plyer import notification
import os
import pandas as pd

khoa_api = " "

# nói

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

# nhận diện giọng nói

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

# xét file lịch trình
file="lichtrinh.xlsx"

# nếu chưa có,tạo file mới
if not os.path.exists(file):
    noi("Hiện chưa có lịch trình, tôi sẽ tạo file mới.")
    df = pd.DataFrame(columns=["tiêu đề", "thời gian"])
    df.to_excel(file, index=False)
    noi("Đã tạo xong")
    
# Danh sách sự kiện trong RAM
def tai_sukien():
    if not os.path.exists(file):
        return []
    df = pd.read_excel(file)
    x = []
    for _, row in df.iterrows():
        try:
            t = pd.to_datetime(row["thời gian"])
            x.append({"tiêu đề": row["tiêu đề"], "thời gian": t})
        except:
            pass
    return x

cacsukien = tai_sukien()
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

        # cập nhật RAM
        global cacsukien
        cacsukien = tai_sukien()

        noi("Đã xoá lịch trình")
    else:
        noi("Không tìm thấy lịch cần xóa")
# hiển thị
def hienthi():
    df=pd.read_excel("lichtrinh.xlsx")
    print(df)
# chạy nền        
def start_reminder():
    import threading
    def chaynen():
        while True:
            hientai = datetime.now()
            for e in cacsukien:
                delta = e["thời gian"] - hientai

                # Nhắc trước 30 phút
                if timedelta(minutes=29) < delta <= timedelta(minutes=30):
                    noi(f"30 phút nữa đến: {e['tiêu đề']} — {e['thời gian']}")

                # Đến giờ
                if timedelta(seconds=0) <= delta <= timedelta(seconds=1):
                    noi(f"ĐẾN GIỜ: {e['tiêu đề']} — {e['thời gian']}")

            time.sleep(60)

    thread = threading.Thread(target=chaynen, daemon=True)
    thread.start()

        
txt = nhangiong()
noi("xin chào")
start_reminder() 
# kiểm tra yêu cầu
while True:
    if "hello" or "xin chào" in txt:
            
        noi("xin hỏi yêu cầu của bạn là gì")
            
        # nói yêu cầu
            
        xt =nhangiong()
        print(xt)
        if 'xóa' in xt:
            xoa()
        elif 'tạm biệt' in xt:
            noi("Chào tạm biệt bạn")
            break
        elif 'hỏi đáp' in xt:
            noi( "Đang khởi động hỏi đáp")
        elif 'xem' in xt:
            noi("Đang in lịch trình")
            hienthi()
        else:

            noi("Xin lỗi,tôi không có chức năng này")           
