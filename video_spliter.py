import sys
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from cutting_video import SplitVideo

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
                    # 첫 프레임 추출 후 이미지로 표시
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
        self.resize(600, 500)
        self.save_directory = ""  # 저장 위치

        # UI 요소 생성
        self.label = DragDropLabel()
        self.label.setMinimumHeight(300)

        self.file_select_button = QPushButton("동영상 파일 선택")
        self.file_select_button.clicked.connect(self.select_file)

        # 동영상 처리 버튼 : 클릭 시 저장 위치를 지정하도록 함.
        self.process_button = QPushButton("동영상 처리")
        self.process_button.clicked.connect(self.process_video)

        # 결과 출력용 라벨 (옵션)
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)

        # 레이아웃 설정
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.label)
        layout.addWidget(self.file_select_button)
        layout.addWidget(self.process_button)
        layout.addWidget(self.result_label)

        self.setCentralWidget(central_widget)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "동영상 선택", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.label.video_file_path = file_path
            # 첫 프레임 추출 후 이미지로 표시
            self.label.show_first_frame(file_path)

    def process_video(self):
        video_path = self.label.get_video_path()
        if not video_path:
            QMessageBox.warning(self, "경고", "먼저 동영상 파일을 선택해주세요.")
            return

        # 동영상 처리 버튼 클릭 시 저장 위치 지정
        directory = QFileDialog.getExistingDirectory(self, "저장 위치 선택", "")
        if not directory:
            QMessageBox.warning(self, "경고", "저장 위치를 지정해주세요.")
            return
        self.save_directory = directory

        # 여기서 동영상 분할 코드를 호출합니다.
        split_video = SplitVideo(video_path, self.save_directory)
        split_video()
        # 처리 완료 후 결과 메시지 출력
        self.result_label.setText(f"동영상 처리 완료!\n저장 위치: {self.save_directory}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
