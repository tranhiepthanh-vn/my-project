# khai báo thư viện và api_key
import time
from datetime import datetime, timedelta
from plyer import notification
import os
import pandas as pd

khoa_api = "AIzaSyC8MBSm2nM7PF2afxzoXH3SwHtxHy_IdOE"

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

# xét file lịch trình
file="lichtrinh.xlsx"
lan_nhac_gan_nhat = {}
# nếu chưa có,tạo file mới
if not os.path.exists(file):
    noi("Hiện chưa có lịch trình, tôi sẽ tạo file mới.")
    df = pd.DataFrame(columns=["tiêu đề", "thời gian"])
    df.to_excel(file, index=False)
    noi("Đã tạo xong")

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
        infer_datetime_format=True
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
                f"{row['tiêu đề']}"
            )
            noi("gần đến giờ thực hiện lịch hôm nay rồi")
            lan_nhac_gan_nhat[key] = now

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
txt=nhangiong()
noi("xin chào")
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
            noi("Đang khởi động hỏi đáp")
        elif 'xem' in xt:
            noi("Đang in lịch trình")
            hienthi()
        else:
            noi("Xin lỗi,tôi không có chức năng này")           
while True:
    kiem_tra_lich()
    time.sleep(60)