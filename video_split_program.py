import sys
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from split_video import SplitVideo

class DragDropLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("여기에 동영상을 드래그하거나 파일을 선택하세요.")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.video_file_path = None  # 업로드된 동영상 파일 경로 저장

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    event.accept()
                    return
        event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    self.video_file_path = file_path
                    self.show_first_frame(file_path)
                    break

    def show_first_frame(self, file_path):
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.setText("첫 프레임을 불러올 수 없습니다.")
        cap.release()

    def get_video_path(self):
        return self.video_file_path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Spliter")
        self.resize(600, 550)
        self.save_directory = ""
        self.split_video = None

        # UI 요소 생성
        self.label = DragDropLabel()
        self.label.setMinimumHeight(300)

        self.file_select_button = QPushButton("동영상 파일 선택")
        self.file_select_button.clicked.connect(self.select_file)

        self.process_button = QPushButton("동영상 처리")
        self.process_button.clicked.connect(self.process_video)

        # 진행 상태를 표시할 ProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)

        # 결과 출력용 라벨
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)

        # 레이아웃 설정
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.label)
        layout.addWidget(self.file_select_button)
        layout.addWidget(self.process_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.result_label)

        self.setCentralWidget(central_widget)

        # OCR 진행 상태 타이머
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)

        # 동영상 자르기 진행 상태 타이머
        self.video_cut_timer = QTimer()
        self.video_cut_timer.timeout.connect(self.update_cut_progress)

        self.cut_video_progress = 0  # 동영상 자르기 진행률

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "동영상 선택", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.label.video_file_path = file_path
            self.label.show_first_frame(file_path)

    def process_video(self):
        video_path = self.label.get_video_path()
        if not video_path:
            QMessageBox.warning(self, "경고", "먼저 동영상 파일을 선택해주세요.")
            return

        directory = QFileDialog.getExistingDirectory(self, "저장 위치 선택", "")
        if not directory:
            QMessageBox.warning(self, "경고", "저장 위치를 지정해주세요.")
            return
        self.save_directory = directory

        # OCR 실행
        self.split_video = SplitVideo(video_path, self.save_directory)
        self.timer.start(500)  # OCR 진행률 업데이트
        self.split_video()  # OCR 실행
        self.timer.stop()  # OCR 완료 후 타이머 정지

        # OCR 완료 후 동영상 자르기 실행
        self.start_cut_video()

    def update_progress(self):
        """OCR 진행 상태 업데이트"""
        if self.split_video:
            task, last = self.split_video.get_task_number()
            if task and last:
                progress = (int(task) / int(last)) * 100
                self.progress_bar.setValue(int(progress))
                self.result_label.setText(f"현재 OCR 진행 중: {task}/{last}")

                if task == last:
                    self.progress_bar.setValue(100)
                    self.timer.stop()

    def start_cut_video(self):
        """OCR 완료 후 동영상 자르기 시작"""
        self.result_label.setText("OCR 완료! 동영상 자르기 시작...")
        self.progress_bar.setValue(0)
        self.cut_video_progress = 0
        self.video_cut_timer.start(500)  # 0.5초마다 진행 상태 업데이트
        self.split_video.cut_video()  # 동영상 자르기 실행
        self.video_cut_timer.stop()  # 완료 후 타이머 정지
        self.result_label.setText("동영상 자르기 완료!")

    def update_cut_progress(self):
        """동영상 자르기 진행 상태 업데이트"""
        if self.split_video:
            total_splits = len(self.split_video.timestamps)
            if total_splits > 0:
                self.cut_video_progress += 100 / total_splits
                if self.cut_video_progress > 100:
                    self.cut_video_progress = 100
                self.progress_bar.setValue(int(self.cut_video_progress))
                self.result_label.setText(f"동영상 자르기 진행 중: {int(self.cut_video_progress)}%")

            if self.cut_video_progress >= 100:
                self.video_cut_timer.stop()
                self.result_label.setText("동영상 자르기 완료!")

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "프로그램 종료", "정말 종료하시겠습니까?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
            print("프로그램 종료")
            sys.exit(0)
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
