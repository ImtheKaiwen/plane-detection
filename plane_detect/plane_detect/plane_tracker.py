import subprocess
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Yalnızca en iyi mp4 video formatını indir

def download_youtube_video(youtube_url, output_path='.'):
    """yt-dlp ile sadece en iyi mp4 video formatını indirir (ses yok)."""
    out_file = os.path.join(output_path, 'video.mp4')
    cmd = [
        'yt-dlp',
        '-f', 'bestvideo[ext=mp4]',
        '-o', out_file,
        youtube_url
    ]
    subprocess.run(cmd, check=True)
    return out_file

if __name__ == '__main__':
    youtube_url = input('YouTube video linkini girin: ')
    video_path = os.path.join(os.path.dirname(__file__), 'video.mp4')
    if not os.path.exists(video_path):
        print('Video indiriliyor...')
        download_youtube_video(youtube_url, output_path=os.path.dirname(__file__))
        print('Video indirildi.')
    else:
        print('video.mp4 zaten mevcut, indirme atlandı.')

    # Videonun başlama saniyesini al
    try:
        start_sec = float(input('Videoyu kaçıncı saniyeden başlatmak istersin? (örn: 0): '))
    except:
        start_sec = 0

    print(f'Video {start_sec}. saniyeden başlatılıyor ve uçaklar tespit ediliyor...')

    model = YOLO('yolov8n.pt')
    # COCO datasetinde uçak class id'si genellikle 4 veya 5 olabilir
    PLANE_CLASS_IDS = [4, 5]

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Her karede YOLO ile uçak tespiti
        results = model(frame)
        boxes = results[0].boxes
        print(f'Tespit edilen kutu sayısı: {len(boxes)}')
        for box in boxes:
            cls = int(box.cls[0])
            class_name = model.model.names[cls] if hasattr(model.model, 'names') else str(cls)
            print(f'Class id: {cls}, Class name: {class_name}')
            if cls in PLANE_CLASS_IDS or class_name.lower() == 'airplane':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 'Plane', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Plane Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() 