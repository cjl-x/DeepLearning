"""
增强版GUI界面 - 集成多种识别模式
支持: 图片识别、摄像头实时识别、视频文件识别
"""
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QInputDialog
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from model import CNN3
from utils import index2emotion, cv2_img_add_text
from blazeface import blaze_detect



class VideoThread(QThread):
    """视频处理线程"""
    change_pixmap_signal = pyqtSignal(np.ndarray, str, list)
    
    def __init__(self, model, source=0):
        super().__init__()
        self.model = model
        self.source = source
        self._run_flag = True
        
    def run(self):
        """运行视频捕获"""
        capture = cv2.VideoCapture(self.source)
        
        while self._run_flag:
            ret, frame = capture.read()
            if ret:
                frame = cv2.resize(frame, (800, 600))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 人脸检测
                faces = blaze_detect(frame_rgb)
                emotion = "未检测到人脸"
                possibility = [0] * 8
                
                if faces is not None and len(faces) > 0:
                    for (x, y, w, h) in faces:
                        # 确保坐标有效
                        x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)
                        if y+h > frame_gray.shape[0]:
                            h = frame_gray.shape[0] - y
                        if x+w > frame_gray.shape[1]:
                            w = frame_gray.shape[1] - x
                        
                        # 检查人脸区域是否有效
                        if w < 10 or h < 10:
                            continue
                            
                        face = frame_gray[y:y+h, x:x+w]
                        
                        # 再次检查提取的人脸是否有效
                        if face.size == 0 or face.shape[0] < 10 or face.shape[1] < 10:
                            continue
                        
                        try:
                            faces_img = self.generate_faces(face)
                            results = self.model.predict(faces_img, verbose=0)
                            result_sum = np.sum(results, axis=0).reshape(-1)
                            label_index = np.argmax(result_sum, axis=0)
                            emotion = index2emotion(label_index)
                            possibility = result_sum.tolist()
                            
                            # 绘制框和文字（文字在右上角）
                            cv2.rectangle(frame_rgb, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 2)
                            text_x = x + w - 80  # 右上角，增加偏移以适应更大字体
                            text_y = y - 20      # 框的上方
                            font_size = max(24, int(h * 0.25))  # 字体大小为框高度的1/4，最小24
                            frame_rgb = cv2_img_add_text(frame_rgb, emotion, text_x, text_y, (255, 255, 255), font_size)
                        except Exception as e:
                            print(f"处理人脸时出错: {e}")
                            continue
                
                self.change_pixmap_signal.emit(frame_rgb, emotion, possibility)
            else:
                break
                
        capture.release()
    
    def generate_faces(self, face_img, img_size=48):
        """生成增广人脸 - 与 recognition_camera.py 保持一致"""
        if face_img is None or face_img.size == 0:
            raise ValueError("Invalid face image")
        
        face_img = face_img / 255.
        face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        resized_images = []
        resized_images.append(face_img)
        resized_images.append(face_img[2:45, :])
        resized_images.append(face_img[1:47, :])
        resized_images.append(cv2.flip(face_img[:, :], 1))
        
        for i in range(len(resized_images)):
            resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
            resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
        
        return np.array(resized_images)
    
    def stop(self):
        """停止线程"""
        self._run_flag = False
        self.wait()


class EnhancedGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = self.load_model()
        self.video_thread = None
        self.init_ui()
        
    def load_model(self):
        """加载CNN模型"""
        model = CNN3()
        model.load_weights('./models/cnn3_best_weights.h5')
        return model
    
    def init_ui(self):
        """初始化UI - 四宫格布局（嵌套布局实现不同行比例）"""
        self.setWindowTitle('人脸表情识别系统')
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建中心部件
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局 - 左右布局（5:5分）
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # ============ 左侧面板（上下3:7分）===========
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        
        # 左上：识别模式
        mode_panel = self.create_mode_panel()
        left_layout.addWidget(mode_panel)
        left_layout.setStretch(0, 3)  
        
        # 左下：视频面板（占70%）
        video_panel = self.create_video_panel()
        left_layout.addWidget(video_panel)
        left_layout.setStretch(1, 7)  
        
        main_layout.addWidget(left_widget)
        main_layout.setStretch(0, 1) 
        
        # ============ 右侧面板（上下6:4分）===========
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        
        # 右上：识别结果（占60%）
        result_panel = self.create_result_panel()
        right_layout.addWidget(result_panel)
        right_layout.setStretch(0, 5)  
        
        # 右下：概率分布（占40%）
        prob_panel = self.create_prob_panel()
        right_layout.addWidget(prob_panel)
        right_layout.setStretch(1, 5)  
        
        main_layout.addWidget(right_widget)
        main_layout.setStretch(1, 1)  
        
    def create_mode_panel(self):
        """创建左上识别模式面板"""
        panel = QtWidgets.QWidget()
        panel.setStyleSheet('border: 2px solid #4CAF50; border-radius: 10px;')
        layout = QtWidgets.QVBoxLayout(panel)
        
        # 标题
        title = QtWidgets.QLabel('识别模式')
        title.setStyleSheet('font-size: 18px; font-weight: bold; padding: 10px; background-color: #4CAF50; color: white;')
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        
        # 模式选择按钮
        self.btn_image = QtWidgets.QPushButton('图片识别')
        self.btn_image.setStyleSheet('padding: 12px; font-size: 15px; margin: 5px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 5px;')
        self.btn_image.clicked.connect(self.image_recognition)
        layout.addWidget(self.btn_image)
        
        self.btn_camera = QtWidgets.QPushButton('摄像头识别')
        self.btn_camera.setStyleSheet('padding: 12px; font-size: 15px; margin: 5px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 5px;')
        self.btn_camera.clicked.connect(self.camera_recognition)
        layout.addWidget(self.btn_camera)
        
        self.btn_video = QtWidgets.QPushButton('视频实时识别')
        self.btn_video.setStyleSheet('padding: 12px; font-size: 15px; margin: 5px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 5px;')
        self.btn_video.clicked.connect(self.video_recognition)
        layout.addWidget(self.btn_video)
        
        # 停止按钮
        self.btn_stop = QtWidgets.QPushButton('停止识别')
        self.btn_stop.setStyleSheet('padding: 12px; font-size: 15px; margin: 5px; background-color: #ff6b6b; color: white; border-radius: 5px;')
        self.btn_stop.clicked.connect(self.stop_recognition)
        self.btn_stop.setEnabled(False)
        layout.addWidget(self.btn_stop)
        
        return panel
    
    def create_result_panel(self):
        """创建右上识别结果面板"""
        panel = QtWidgets.QWidget()
        panel.setStyleSheet('border: 2px solid #2196F3; border-radius: 10px;')
        layout = QtWidgets.QVBoxLayout(panel)
        
        # 标题（占2份）
        title = QtWidgets.QLabel('识别结果')
        title.setStyleSheet('font-size: 30px; font-weight: bold; padding: 10px; background-color: #2196F3; color: white;')
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        layout.setStretch(0, 2)  
        
        # 表情显示（占3份）
        self.label_emotion = QtWidgets.QLabel('表情: 未识别')
        self.label_emotion.setStyleSheet('font-size: 30px; padding: 15px; font-weight: bold;')
        self.label_emotion.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.label_emotion)
        layout.setStretch(1, 3)  
        
        # 表情图标（占5份）
        icon_container = QtWidgets.QWidget()
        icon_layout = QtWidgets.QVBoxLayout(icon_container)
        self.label_icon = QtWidgets.QLabel()
        self.label_icon.setFixedSize(250, 250)
        self.label_icon.setAlignment(QtCore.Qt.AlignCenter)
        self.label_icon.setStyleSheet('border: 2px solid #2196F3; border-radius: 15px;')
        icon_layout.addWidget(self.label_icon, alignment=QtCore.Qt.AlignCenter)
        icon_layout.addStretch()
        layout.addWidget(icon_container)
        layout.setStretch(2, 5)  
        
    
        
        return panel
    
    def create_video_panel(self):
        """创建左下视频面板（实时画面+视频分析标签页）"""
        panel = QtWidgets.QWidget()
        panel.setStyleSheet('border: 2px solid #ff9800; border-radius: 10px;')
        layout = QtWidgets.QVBoxLayout(panel)
        
        # 创建标签页
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setStyleSheet('''
            QTabBar::tab { padding: 10px 25px; font-size: 14px; font-weight: bold; }
            QTabWidget::pane { border: none; background: transparent; }
            QTabWidget::tab-bar { border: none; }
        ''')
        
        # 标签页1: 实时画面
        self.realtime_tab = self.create_realtime_tab()
        self.tab_widget.addTab(self.realtime_tab, '实时画面')
        
        layout.addWidget(self.tab_widget)
        
        return panel
    
    def create_prob_panel(self):
        """创建右下概率分布面板"""
        panel = QtWidgets.QWidget()
        panel.setStyleSheet('border: 2px solid #9c27b0; border-radius: 10px;')
        layout = QtWidgets.QVBoxLayout(panel)
        
        # 标题（占1份）
        title = QtWidgets.QLabel('概率分布')
        title.setStyleSheet('font-size: 30px; font-weight: bold; padding: 10px; background-color: #9c27b0; color: white;')
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        layout.setStretch(0, 1)  # 标题占1份
        
        # 概率分布图（占5份）
        self.figure = Figure(figsize=(6, 4), dpi=80)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        layout.setStretch(1, 5)  # 图表占5份
        
        return panel
    
    def create_realtime_tab(self):
        """创建实时画面标签页"""
        tab = QtWidgets.QWidget()
        tab.setStyleSheet('border: none; background: transparent;')
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 标题
        title = QtWidgets.QLabel('实时画面')
        title.setStyleSheet('font-size: 18px; font-weight: bold;')
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        
        # 图像显示区域（无边框）
        self.image_label = QtWidgets.QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet('background-color: #f8f9fa; border: none;')
        self.image_label.setText('请选择识别模式')
        layout.addWidget(self.image_label)
        
        # 状态栏（无边框）
        self.status_label = QtWidgets.QLabel('就绪')
        self.status_label.setStyleSheet('padding: 5px; background-color: #e9ecef; border: none;')
        layout.addWidget(self.status_label)
        
        return tab
    

    
    def image_recognition(self):
        """图片识别模式"""
        # 检查是否有实时识别正在进行
        if self.video_thread is not None and self.video_thread.isRunning():
            QMessageBox.warning(self, '提示', '请先停止当前的实时识别！')
            return
        
        file_name, _ = QFileDialog.getOpenFileName(
            self, '选择图片', './input/test/',
            'Images (*.png *.jpg *.jpeg *.bmp)'
        )
        
        if file_name:
            self.status_label.setText(f'正在识别: {os.path.basename(file_name)}')
            
            # 读取并显示原图
            img = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 人脸检测
            faces = blaze_detect(img_rgb)
            
            if faces is None or len(faces) == 0:
                QMessageBox.warning(self, '警告', '未检测到人脸!')
                self.status_label.setText('未检测到人脸')
                return
            
            # 识别表情
            emotion = "未知"
            possibility = [0] * 8
            
            for (x, y, w, h) in faces:
                # 确保坐标有效
                x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)
                if y+h > img_gray.shape[0]:
                    h = img_gray.shape[0] - y
                if x+w > img_gray.shape[1]:
                    w = img_gray.shape[1] - x
                
                # 检查人脸区域是否有效
                if w < 10 or h < 10:
                    continue
                
                face = img_gray[y:y+h, x:x+w]
                
                # 检查提取的人脸是否有效
                if face.size == 0 or face.shape[0] < 10 or face.shape[1] < 10:
                    continue
                
                try:
                    faces_img = self.generate_faces(face)
                    results = self.model.predict(faces_img, verbose=0)
                    result_sum = np.sum(results, axis=0).reshape(-1)
                    label_index = np.argmax(result_sum, axis=0)
                    emotion = index2emotion(label_index)
                    possibility = result_sum.tolist()
                    
                    # 绘制框和文字（文字在右上角）
                    cv2.rectangle(img_rgb, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 2)
                    text_x = x + w - 80  # 右上角，增加偏移以适应更大字体
                    text_y = y - 20      # 框的上方
                    font_size = max(24, int(h * 0.25))  # 字体大小为框高度的1/4，最小24
                    img_rgb = cv2_img_add_text(img_rgb, emotion, text_x, text_y, (255, 255, 255), font_size)
                except Exception as e:
                    print(f"处理人脸时出错: {e}")
                    continue
            
            # 显示结果
            self.display_image(img_rgb)
            self.update_result(emotion, possibility)
            
            # 保存结果
            if not os.path.exists('./output'):
                os.makedirs('./output')
            cv2.imwrite('./output/rst.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            self.status_label.setText(f'识别完成: {emotion} (已保存到 output/rst.png)')
    
    def camera_recognition(self):
        """摄像头识别模式"""
        if self.video_thread is not None and self.video_thread.isRunning():
            QMessageBox.warning(self, '警告', '请先停止当前识别!')
            return
        
        self.status_label.setText('正在启动摄像头...')
        self.video_thread = VideoThread(self.model, source=0)
        self.video_thread.change_pixmap_signal.connect(self.update_video_frame)
        self.video_thread.start()
        
        # 禁用其他实时识别模式按钮（不包括视频异步分析）
        self.btn_stop.setEnabled(True)
        self.btn_image.setEnabled(False)
        self.btn_camera.setEnabled(False)
        self.btn_video.setEnabled(False)
        self.status_label.setText('摄像头识别中... (按ESC或点击停止按钮退出)')
    
    def video_recognition(self):
        """视频文件识别模式"""
        if self.video_thread is not None and self.video_thread.isRunning():
            QMessageBox.warning(self, '警告', '请先停止当前识别!')
            return
        
        file_name, _ = QFileDialog.getOpenFileName(
            self, '选择视频文件', './',
            'Videos (*.mp4 *.avi *.mov *.mkv)'
        )
        
        if file_name:
            self.status_label.setText(f'正在处理视频: {os.path.basename(file_name)}')
            self.video_thread = VideoThread(self.model, source=file_name)
            self.video_thread.change_pixmap_signal.connect(self.update_video_frame)
            self.video_thread.start()
            
            # 禁用其他实时识别模式按钮（不包括视频异步分析）
            self.btn_stop.setEnabled(True)
            self.btn_image.setEnabled(False)
            self.btn_camera.setEnabled(False)
            self.btn_video.setEnabled(False)
    
    def stop_recognition(self):
        """停止识别"""
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread = None
        
        # 恢复实时识别按钮（不包括视频异步分析）
        self.btn_stop.setEnabled(False)
        self.btn_image.setEnabled(True)
        self.btn_camera.setEnabled(True)
        self.btn_video.setEnabled(True)
        self.status_label.setText('已停止识别')
        self.image_label.setText('请选择识别模式')
    
    def update_video_frame(self, frame, emotion, possibility):
        """更新视频帧"""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ))
        
        self.update_result(emotion, possibility)
    
    def display_image(self, img_rgb):
        """显示图片"""
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ))
    
    def update_result(self, emotion, possibility):
        """更新识别结果"""
        self.label_emotion.setText(f'表情: {emotion}')
        
        # 显示表情图标
        if emotion != '未检测到人脸' and emotion != '未知':
            icon_path = f'./assets/icons/{emotion}.png'
            if os.path.exists(icon_path):
                pixmap = QtGui.QPixmap(icon_path)
                self.label_icon.setPixmap(pixmap.scaled(
                    100, 100, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
                ))
        
        # 更新概率分布图
        self.update_probability_chart(possibility)
    
    def update_probability_chart(self, possibility):
        """更新概率分布图"""
        self.ax.clear()
        emotions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        colors = ['#ff6b6b', '#95e1d3', '#f38181', '#ffd93d', '#6c5ce7', '#a29bfe', '#74b9ff', '#fd79a8']
        
        bars = self.ax.bar(emotions, possibility, color=colors, alpha=0.7)
        self.ax.set_ylabel('概率', fontproperties='SimHei')
        self.ax.set_title('表情概率分布', fontproperties='SimHei')
        self.ax.set_ylim([0, max(possibility) * 1.2 if max(possibility) > 0 else 1])
        
        # 旋转x轴标签
        self.ax.tick_params(axis='x', rotation=45)
        
        # 在柱子上显示数值
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def generate_faces(self, face_img, img_size=48):
        """生成增广人脸 - 与 recognition_camera.py 保持一致"""
        if face_img is None or face_img.size == 0:
            raise ValueError("Invalid face image")
        
        face_img = face_img / 255.
        face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        resized_images = []
        resized_images.append(face_img)
        resized_images.append(face_img[2:45, :])
        resized_images.append(face_img[1:47, :])
        resized_images.append(cv2.flip(face_img[:, :], 1))
        
        for i in range(len(resized_images)):
            resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
            resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
        
        return np.array(resized_images)
    
    def keyPressEvent(self, event):
        """键盘事件"""
        if event.key() == QtCore.Qt.Key_Escape:
            self.stop_recognition()
    
    def closeEvent(self, event):
        """关闭事件"""
        self.stop_recognition()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建并显示主窗口
    window = EnhancedGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()