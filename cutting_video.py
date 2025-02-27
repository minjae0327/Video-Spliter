import re
import cv2
import easyocr

reader = easyocr.Reader(['en'])

class SplitVideo():
    def __init__(self, video_path, save_path):
        self.video_path = video_path
        self.save_path = save_path
        self.previous_text = ""
        self.timestamps = []
        self.last_inst = None
        
    def __call__(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        roi = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1980, 1080))

            if roi is None:
                roi = cv2.selectROI("Select ROI", frame, False, False)
                cv2.destroyWindow("Select ROI")
                x, y, w, h = roi

            if frame_count % 6 == 0:
                cropped_frame = frame[y:y+h, x:x+w]

                # OCR 모델 예측
                inst_number, last_inst = self.extract_task_number(cropped_frame)

                if inst_number is None or last_inst is None:
                    #print("OCR 실패: 유효한 작업 번호를 찾을 수 없음")
                    continue

                if self.last_inst is None:
                    self.last_inst = last_inst
                    print(f"last_inst 검출됨: {self.last_inst}")

                if inst_number == self.last_inst:
                    print("OCR 종료")
                    break

                # 현재 동영상 시간(ms) 가져오기
                timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

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
        text_list = reader.readtext(frame, detail=0)

        # OCR 결과가 없을 경우 예외 처리
        if not text_list:
            return None, None

        # OCR 결과 문자열 결합
        text = " ".join(text_list)

        # 괄호 안의 두 숫자를 각각 캡처하는 정규식
        match = re.search(r'\((\d+)/(\d+)\)', text)
        if match:
            task_number = match.group(1)
            last_instruction = match.group(2)
            return task_number, last_instruction

        return None, None  # OCR이 정상적으로 수행되지 않은 경우 예외 처리
