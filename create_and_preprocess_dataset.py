"""
데이터셋 생성 및 전처리 코드

"""

import cv2
import os
import torch
import evaluate
from torchvision import transforms
from datasets import Dataset
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image, ImageOps
import numpy as np

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class create_dataset():
    def __init__(self):
        self.roi = None
        self.cropping = False
        self.start_point = None
        self.img_copy = None
        
        
    def mouse_crop(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.cropping = True
            start_point = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.cropping = False
            end_point = (x, y)
            x1, y1 = start_point
            x2, y2 = end_point
            
            self.roi = (min(x1,x2), min(y1,y2), abs(x2 - x1), abs(y2 - y1))
            cv2.rectangle(self.img_copy, start_point, end_point, (0,255,0), 2)
            cv2.imshow("Select ROI", self.img_copy)
            
            
    def select_rot_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise Exception("영상에서 첫 프레임을 읽을 수 없습니다.")
        
        frame = cv2.resize(frame, (1920, 1080))
        self.img_copy = frame.copy()
        cv2.imshow("Select ROI", self.img_copy)
        cv2.setMouseCallback("Select ROI", self.mouse_crop)
        print("영역을 드래그하세요. 완료 후 아무 키나 누르세요.")
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return self.roi
    
    
    def ocr_infer(self, image):
        try:
            pil_image = Image.fromarray(image).convert("RGB")
            pixel_values = processor(image=pil_image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch
        except Exception as e:
            print(f"OCR 실패: {e}")
            
            return None
        
    
    def process_video(self, video_path, save_root, interval):
        roi_box = self.select_rot_from_video(video_path)
        
        if roi_box:
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            
            os.makedirs(save_root, exist_ok=True)
            
            etc_path = os.path.join(save_root, "ETC")
            os.makedirs(etc_path, exist_ok=True) 
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % interval == 0:
                    frame = cv2.resize(frame, (1920, 1080))
                    x, y, w, h = roi_box
                    cropped = frame[y:y+h, x:x+w]
                    label = self.ocr_infer(cropped)
                    label = label.replace("/", "_")
                    
                    # 폴더 분류 저장
                    if label and len(label) > 0 and all(c.isprintable() for c in label):
                        folder_name = os.path.join(save_root, label)
                    else:
                        folder_name = etc_path
                        
                    os.makedirs(folder_name, exist_ok=True)
                    save_path = os.paht.join(folder_name, f"frame_{frame_idx:05}.jpg")
                    cv2.imwrite(save_path, cropped)
                    print(f"{frame_idx} 저장 → {folder_name}")
                
                frame_idx += 1

            cap.release()
            
        else:
            print("프로그램 종료")
            


class preprocess():
    def __init__(self):
        pass
    
    
    def __call__(self, image_dir):
        self.balalce(image_dir)
        
        data = self.load_data(image_dir)
        dataset = Dataset.from_pandas(data)
    
        processed_dataset = dataset.map(self.preprocess)
        processed_dataset = processed_dataset.remove_columns(["label", "image_path"])

        split_dataset = processed_dataset.train_test_split(test_size=0.2)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
        
        return train_dataset, test_dataset
    
    
    #클래스 불균형 방지
    def balalce(self, image_dir):
        labels = os.listdir(image_dir)
        
        label_length = {}
        count = float("inf")
        
        for label in labels:
            if label == "ETC":
                continue
            label_path = os.path.join(image_dir, label)
    
            files = os.listdir(label_path)
            label_length[label] = files

            if count > len(files):
                count = len(files)

        print("가장 적은 프레임 수:", count)
        
        for label, files in label_length.items():
            label_path = os.path.join(image_dir, label)
            
            if len(files) > count:
                for delete_file in files[count:]:
                    os.remove(os.path.join(label_path, delete_file))
                    
                    
    def load_data(image_root):
        records = []
        
        for label in os.listdir(image_root):
            label_path = os.path.join(image_root, label)
            for filename in os.listdir(label_path):
                if filename.endswith((".jpg", ".png")):
                    records.append({
                        "image_path": os.path.join(label_path, filename),
                        "label": label.replace("_", "/")
                    })
                    
        return pd.DataFrame(records)
    
    
    def pad_to_square(self, image, target_size=384):
        w, h = image.size
        scale = min(target_size / w, target_size / h)
        resized = image.resize((int(w * scale), int(h * scale)))
        padded = ImageOps.pad(resized, (target_size, target_size), color=(0, 0, 0))
        
        return padded


    def preprocess(self, data):
        image = cv2.imread(data['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        image = Image.fromarray(binary_image).convert("RGB")
        
        image = self.pad_to_square(image, target_size=384)
        pixel_values = processor(images=image, return_tensors="pt").pixel_values[0]
        labels = processor.tokenizer(data['label'], padding="max_length", max_length=10, truncation=True).input_ids
        labels = [l if l != processor.tokenizer.pad_token_id else -100 for l in labels]
        
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels)
        }
        