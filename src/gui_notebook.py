import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

DEFAULT_EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']


def _to_rgb(img, color_format='bgr'):
    """Convert BGR or gray image to RGB for notebook display."""
    if img is None:
        raise ValueError('result_img cannot be None')

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.ndim == 3 and img.shape[2] == 3:
        if color_format == 'bgr':
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if color_format == 'rgb':
            return img

    return img


def display_notebook_page(result_img, emotions, probabilities, EMOTIONS=None, face_index=0, title='表情识别结果', color_format='bgr'):
    """Display a notebook page similar to gui_enhanced.py.

    Parameters
    ----------
    result_img : np.ndarray
        The image with face boxes/labels (BGR or RGB depending on color_format).
    emotions : list[str]
        The predicted emotion names for each detected face.
    probabilities : list[list[float]]
        The probability distribution for each face.
    EMOTIONS : list[str], optional
        The emotion labels corresponding to probability values.
    face_index : int, default 0
        Which face's probability distribution to show.
    title : str, default '表情识别结果'
        The page title.
    color_format : str, default 'bgr'
        The color format of result_img. Use 'rgb' if the image is already RGB.
    """
    if EMOTIONS is None:
        EMOTIONS = DEFAULT_EMOTIONS

    if face_index < 0 or face_index >= len(emotions):
        raise IndexError('face_index out of range')

    result_rgb = _to_rgb(result_img, color_format=color_format)
    prob = probabilities[face_index]
    emotion = emotions[face_index]

    fig = plt.figure(figsize=(16, 9))
    grid = plt.GridSpec(4, 6, figure=fig, wspace=0.4, hspace=0.4)

    ax_image = fig.add_subplot(grid[:, :3])
    ax_header = fig.add_subplot(grid[0, 3:])
    ax_text = fig.add_subplot(grid[1:2, 3:])
    ax_bar = fig.add_subplot(grid[2:, 3:])

    ax_image.imshow(result_rgb)
    ax_image.axis('off')
    font_path = os.path.join(os.path.dirname(__file__), '../assets/simsun.ttc')
    font_path = os.path.abspath(font_path)
    font = FontProperties(fname=font_path)

    ax_image.set_title('识别结果图像', fontsize=20, pad=16, fontproperties=font)

    ax_header.axis('off')
    ax_header.text(0.5, 0.5, title, ha='center', va='center', fontsize=24, fontweight='bold', fontproperties=font)

    ax_text.axis('off')
    ax_text.text(0.5, 0.7, f'表情: {emotion}', ha='center', va='center', fontsize=22, fontproperties=font)
    ax_text.text(0.5, 0.35, f'人脸 {face_index + 1} / {len(emotions)}', ha='center', va='center', fontsize=16, fontproperties=font)

    colors = ['#ff6b6b', '#95e1d3', '#f38181', '#ffd93d', '#6c5ce7', '#a29bfe', '#74b9ff', '#fd79a8']
    x_positions = np.arange(len(EMOTIONS))
    bars = ax_bar.bar(x_positions, prob, color=colors[: len(EMOTIONS)], alpha=0.85)
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(EMOTIONS, fontproperties=font)
    ax_bar.set_ylim(0, max(max(prob), 1.0) * 1.2)
    ax_bar.set_ylabel('概率', fontsize=14, fontproperties=font)
    ax_bar.set_title('表情概率分布', fontsize=18, pad=12, fontproperties=font)
    ax_bar.tick_params(axis='x', rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2.0,
                    height + 0.02,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)

    fig.subplots_adjust(top=0.95, bottom=0.08, left=0.05, right=0.98, hspace=0.35, wspace=0.35)
    plt.show()


def display_prediction_page(image_path, model, predictor, EMOTIONS=None, face_index=0, title='表情识别结果', color_format='bgr'):
    """Run prediction and display a notebook page."""
    result_img, emotions, probabilities = predictor(image_path, model)
    display_notebook_page(result_img, emotions, probabilities, EMOTIONS=EMOTIONS, face_index=face_index, title=title, color_format=color_format)
    return result_img, emotions, probabilities


def display_prediction_pages(image_paths, model, predictor, EMOTIONS=None, title_prefix='人脸表情识别结果', color_format='bgr', return_results=False):
    """Run prediction and display notebook pages for multiple images."""
    results = []
    for i, image_path in enumerate(image_paths):
        title = f"{title_prefix} ({i+1}/{len(image_paths)})"
        result_img, emotions, probabilities = predictor(image_path, model)
        display_notebook_page(result_img, emotions, probabilities, EMOTIONS=EMOTIONS, face_index=0, title=title, color_format=color_format)
        results.append((image_path, result_img, emotions, probabilities))
    if return_results:
        return results
    return None
