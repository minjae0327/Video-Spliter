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
        
        
    def mouse_crop(self, event, x, y, flags, param):
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
        
        #
            
            
            
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
            