"""
纯逻辑模块：模型加载、图像检测、表情预测、结果标注
"""
from pathlib import Path
import os
import cv2
import numpy as np

from model import CNN3
from utils import index2emotion, cv2_img_add_text
from blazeface import blaze_detect

EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_WEIGHTS_PATH = ROOT_DIR / 'models' / 'cnn3_best_weights.h5'
HAAR_CASCADE_PATH = ROOT_DIR / 'dataset' / 'params' / 'haarcascade_frontalface_alt.xml'


def load_model(weights_path: str = None):
    """加载 CNN3 模型权重"""
    if weights_path is None:
        weights_path = str(MODEL_WEIGHTS_PATH)
    model = CNN3()
    model.load_weights(weights_path)
    return model


def read_image(image_path):
    """读取图片，支持中文路径"""
    image_path = str(image_path)
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    return img


def detect_faces(img, method='blazeface'):
    """检测图片中的人脸框"""
    if method == 'blazeface':
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = blaze_detect(img_rgb)
        return [] if faces is None else faces

    if method == 'haar':
        face_cascade = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(30, 30)
        )
        return [] if faces is None else faces

    raise ValueError(f"不支持的检测方法: {method}")


def prepare_face_images(face_gray, img_size=48):
    """将单个人脸灰度图增强为多个输入样本"""
    if face_gray is None or face_gray.size == 0:
        raise ValueError("Invalid face image")

    face = face_gray / 255.0
    face = cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    patches = [
        face,
        face[2:45, :],
        face[1:47, :],
        cv2.flip(face, 1)
    ]

    faces = []
    for patch in patches:
        patch = cv2.resize(patch, (img_size, img_size))
        faces.append(np.expand_dims(patch, axis=-1))

    return np.array(faces)


def predict_face(face_gray, model):
    """预测单个人脸的表情概率分布"""
    faces = prepare_face_images(face_gray)
    results = model.predict(faces, verbose=0)
    result_sum = np.sum(results, axis=0).reshape(-1)
    label_index = int(np.argmax(result_sum, axis=0))
    emotion = index2emotion(label_index, kind='en')
    return emotion, result_sum.tolist(), label_index


def annotate_image(img, face_bbox, emotion, color=(0, 255, 0), font_size=20):
    """在图像上绘制人脸框和表情文字"""
    x, y, w, h = face_bbox
    x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)
    cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
    img = cv2_img_add_text(img, emotion, x + 30, y + 30, (255, 255, 255), font_size)
    return img


def predict_image_path(image_path, model=None, detector='blazeface', save_path=None):
    """对图片路径进行人脸检测和表情预测，返回标注图像和识别结果"""
    img = read_image(image_path)
    if model is None:
        model = load_model()

    faces = detect_faces(img, method=detector)
    if len(faces) == 0:
        return img, [], []

    emotions = []
    possibilities = []
    for face in faces:
        x, y, w, h = face
        x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)
        x2 = min(x + w + 10, img.shape[1])
        y2 = min(y + h + 10, img.shape[0])
        face_gray = cv2.cvtColor(img[y:y2, x:x2], cv2.COLOR_BGR2GRAY)

        if face_gray.size == 0:
            continue

        emotion, possibility, _ = predict_face(face_gray, model)
        emotions.append(emotion)
        possibilities.append(possibility)
        img = annotate_image(img, (x, y, w, h), emotion)

    if save_path is not None:
        save_path = str(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)

    return img, emotions, possibilities


def predict_image_array(img, model=None, detector='blazeface'):
    """对已加载的 BGR 图像进行人脸检测和表情预测"""
    if model is None:
        model = load_model()

    faces = detect_faces(img, method=detector)
    if len(faces) == 0:
        return img, [], []

    emotions = []
    possibilities = []
    for face in faces:
        x, y, w, h = face
        x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)
        x2 = min(x + w + 10, img.shape[1])
        y2 = min(y + h + 10, img.shape[0])
        face_gray = cv2.cvtColor(img[y:y2, x:x2], cv2.COLOR_BGR2GRAY)

        if face_gray.size == 0:
            continue

        emotion, possibility, _ = predict_face(face_gray, model)
        emotions.append(emotion)
        possibilities.append(possibility)
        img = annotate_image(img, (x, y, w, h), emotion)

    return img, emotions, possibilities
