import cv2
import os
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np

# 1. TroCR 모델 준비
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. 마우스로 ROI 선택
roi = None
cropping = False
start_point = None

def mouse_crop(event, x, y, flags, param):
    global roi, cropping, start_point, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        cropping = True
        start_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        cropping = False
        end_point = (x, y)
        x1, y1 = start_point
        x2, y2 = end_point
        roi = (min(x1,x2), min(y1,y2), abs(x2 - x1), abs(y2 - y1))
        cv2.rectangle(img_copy, start_point, end_point, (0,255,0), 2)
        cv2.imshow("Select ROI", img_copy)

def select_roi_from_video(video_path):
    global img_copy
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise Exception("영상에서 첫 프레임을 읽을 수 없습니다.")
    
    img_copy = frame.copy()
    cv2.imshow("Select ROI", img_copy)
    cv2.setMouseCallback("Select ROI", mouse_crop)
    print("▶ ROI 영역을 드래그하세요. 완료 후 아무 키나 누르세요.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return roi

# 3. OCR 수행 함수
def ocr_infer(image):
    try:
        pil_image = Image.fromarray(image).convert("RGB")
        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()
    except Exception as e:
        print(f"OCR 실패: {e}")
        return None

# 4. 영상 처리 및 프레임 저장
def process_video(video_path, save_root):
    roi_box = select_roi_from_video(video_path)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    etc_path = os.path.join(save_root, "ETC")
    os.makedirs(etc_path, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1920, 1080))

        x, y, w, h = roi_box
        cropped = frame[y:y+h, x:x+w]
        label = ocr_infer(cropped)

        # 폴더 분류 저장
        if label and len(label) > 0 and all(c.isprintable() for c in label):
            folder_name = os.path.join(save_root, label)
        else:
            folder_name = etc_path

        os.makedirs(folder_name, exist_ok=True)
        save_path = os.path.join(folder_name, f"frame_{frame_idx:05}.jpg")
        cv2.imwrite(save_path, cropped)
        print(f"✔ Frame {frame_idx} 저장 → {folder_name}")
        frame_idx += 1

    cap.release()

# 실행
if __name__ == "__main__":
    video_path = "dataset\Installation of the MLG Hydraulic Dressings.mp4"  # 🔁 여기에 비디오 경로 설정
    save_directory = "output_frames"
    process_video(video_path, save_directory)
