import cv2
import numpy as np
import gradio as gr
import requests
import os
import time

file_urls = [
    "https://www.dropbox.com/scl/fi/w6rc4prpemzfmsuybznf5/pedestrians.mp4?rlkey=0mr0frqzo0vzs81zvb5wabb2l&st=6lagqxv9&dl=0"
]

def download_url(url, save_name):
    url = url
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)

for i, url in enumerate(file_urls):
    if 'mp4' in file_urls[i]:
        download_url(
            file_urls[i],
            f"video.mp4"
        )
video_path = [['video.mp4']]

def ids(video):
    scale = 0.4
    video = cv2.VideoCapture(video)
    if not video.isOpened():
        print("❌ Error: Could not open video file")
        exit()

    kernel = None


    bg_obj = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)*3)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    w1 = int(width*scale)
    h1 = int(height*3*scale)

    target_fps = 3
    delay = 1.0 / target_fps

    while True:
        start_time = time.time()
        ret, frame = video.read()

        if not ret:
            break

        fgmask = bg_obj.apply(frame)

        _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frameCopy = frame.copy()

        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frameCopy, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frameCopy, 'Object Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (0, 255, 0), 1, cv2.LINE_AA)
        # fg_part = cv2.bitwise_and(frame, frame, mask=fgmask)
        fg_part = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2RGB)
        stack = np.hstack((fg_part, frameCopy))

        stack_rv = cv2.resize(stack, (w1,h1))

        yield stack_rv #frame, fg_part, frameCopy

        elapsed = time.time() - start_time
        if elapsed < delay:
            time.sleep(delay - elapsed)


inputs = gr.Video(label="Upload Video")
outputs = gr.Image(type="numpy", label="Processed Frame")

interface = gr.Interface(
    fn=ids,
    inputs=inputs,
    outputs=outputs,
    title="Object Detection Stream",
    examples=video_path,
    description="Processes each frame of the video and highlights detected motion objects.",
    cache_examples=False
)

interface.launch()


