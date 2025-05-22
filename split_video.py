import os
import re
import cv2
import ffmpeg
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class SplitVideo():
    def __init__(self, video_path, save_path):
        self.video_path =video_path
        self.save_path = save_path
        self.previous_text = ""
        self.timestamps = []
        self.last_inst = None
        self.task_number = 0
        self.last_instruction = 0
        self.processor = processor
        self.model = model
        
        
    def __call__(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        roi = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (960, 540))

            if roi is None:
                roi = cv2.selectROI("Select ROI", frame, False, False)
                cv2.destroyWindow("Select ROI")
                x, y, w, h = roi

            if frame_count % 30 == 0:
                cropped_frame = frame[y:y+h, x:x+w]

                # OCR 모델 예측
                inst_number, last_inst = self.extract_task_number(cropped_frame)

                if inst_number is None or last_inst is None:
                    #print("OCR 실패: 유효한 작업 번호를 찾을 수 없음")
                    continue

                if self.last_inst is None:
                    self.last_inst = last_inst
                    print(f"last_inst 검출됨: {self.last_inst}")
                    
                # 현재 동영상 시간(ms) 가져오기
                timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

                if inst_number == self.last_inst:
                    self.record_timestemp(inst_number, timestamp)
                    print("OCR 종료")
                    break

                # OCR 결과 비교 및 시간 기록
                self.record_timestemp(inst_number, timestamp)

            frame_count += 1

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
    
    # OCR 결과가 이전과 다를 경우 변경된 시점 저장
    def record_timestemp(self, predicted_text, timestamp):
        # OCR 결과가 이전과 다를 경우
        if 1 <= int(predicted_text) <= 99 and predicted_text != self.previous_text:
            print(f"[{timestamp} ms] OCR 변경 감지: {predicted_text}")
            self.timestamps.append(timestamp)
            self.previous_text = predicted_text  # 이전 텍스트 업데이트
            

    def extract_task_number(self, frame):
        # 1. OpenCV 프레임 → PIL 이미지로 변환
        image = Image.fromarray(frame).convert("RGB")

        # 2. TrOCR 전처리
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.model.device)

        # 3. 모델 추론
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_length=10)

        # 4. 디코딩
        generated_text = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # 5. 정규식으로 작업번호 추출
        match = re.search(r'\((\d+)/(\d+)\)', generated_text)
        if match:
            self.task_number = match.group(1)
            self.last_instruction = match.group(2)
            return self.task_number, self.last_instruction

        return None, None  # 인식 실패 시
    
    
    def get_video_duration(self, video_path):
        probe = ffmpeg.probe(video_path)
        duration_sec = float(probe['format']['duration'])
        return duration_sec
    
    
    def cut_video(self):
        if not self.timestamps:
            print("잘라낼 타임스탬프가 없습니다.")
            return

        # 원본 파일명 가져오기
        original_filename = os.path.splitext(os.path.basename(self.video_path))[0]
        
        # 저장 폴더 생성
        save_folder_path = os.makedirs(self.save_path + "/" + original_filename, exist_ok=True)
        os.makedirs(save_folder_path, exist_ok=True)

        # 원본 영상 정보
        cap = cv2.VideoCapture(self.video_path)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        total_sec = self.get_video_duration(self.video_path)

        # 타임스탬프를 기준으로 분할 수행
        for idx, start_time in enumerate(self.timestamps):
            end_time = self.timestamps[idx + 1] if idx + 1 < len(self.timestamps) else None

            # 초 단위로 변환 (OpenCV에서는 ms 단위이므로 1000으로 나눔)
            start_sec = start_time / 1000
            end_sec = end_time / 1000 if end_time else total_sec

            # 저장될 파일명 설정
            output_filename = f"{original_filename}_{idx+1}.mp4"
            output_path = os.path.join(save_folder_path, output_filename)

            # ffmpeg 명령어 설정
            ffmpeg_cmd = (
                ffmpeg
                .input(self.video_path, ss=start_sec, to=end_sec if end_sec else None)
                .output(output_path, c="copy")
                .overwrite_output()
            )

            # 실행
            try:
                ffmpeg_cmd.run()
                print(f"저장 완료: {output_path}")
            except ffmpeg.Error as e:
                print(f"ffmpeg 오류 발생: {e}")
    
    
    def get_task_number(self):
        return self.task_number, self.last_instruction

# if __name__ == '__main__':
#     video_path = "C://Users//minja//Downloads//folder//Installation of the MLG Disconnection Box.mp4"
#     save_path = "C://Users//minja//Downloads//folder"
#     aa = SplitVideo(video_path, save_path)
#     aa.cut_video()