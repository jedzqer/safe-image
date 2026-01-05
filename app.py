# -*- coding: utf-8 -*-
"""
一个基于 Flask 的图像处理 Web 应用。

核心功能:
- 提供一个 Web 界面，用于根据用户选择的标签和效果来处理图像。
- 动态地对图像中的特定区域（由 JSON 文件标注）应用多种视觉效果（如马赛克、模糊等）。
- 支持复杂的图像筛选逻辑，包括必选标签、排除标签和随机标签。
- 提供 API 用于删除图像及其关联的标注文件。
- 内置性能优化，如 JSON 标注文件预加载缓存和并发处理。
"""

import os
import glob
import random
import json
import base64
import cv2
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union, Any
import datetime
from collections import deque
import torch
import faiss
from cn_clip.clip import load_from_name, tokenize
from PIL import Image, UnidentifiedImageError
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, flash, Response
)
from werkzeug.utils import secure_filename

# --- 语义搜索组件 ---

# 全局变量定义
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"语义搜索推理设备: {device}")

INDEX_FILE = "image_features.index"
METADATA_FILE = "index_metadata.json"

# 初始化全局变量
search_model = None  # type: Any
search_index = None  # type: Any
search_image_map = None  # type: Optional[Dict[str, str]] 
SEARCH_COMPONENTS_LOADED = False

try:
    print("正在加载 CLIP 模型...")
    search_model, _ = load_from_name("ViT-L-14", device=device)
    search_model.eval()

    if os.path.isfile(INDEX_FILE) and os.path.isfile(METADATA_FILE):
        try:
            print(f"正在从 {INDEX_FILE} 加载索引...")
            search_index = faiss.read_index(INDEX_FILE)

            print(f"正在从 {METADATA_FILE} 加载元数据...")
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                search_image_map = metadata.get("image_map", {})
            
            if not search_image_map:
                raise ValueError("元数据中的 'image_map' 为空或不存在。")

            SEARCH_COMPONENTS_LOADED = True
            print("语义搜索组件加载成功")

        except Exception as e:
            print(f"警告：加载索引或元数据文件时出错: {e}")
            search_model = None
            search_index = None
            search_image_map = None
            SEARCH_COMPONENTS_LOADED = False
    else:
        print("警告：找不到索引文件或元数据文件，语义搜索功能将被禁用")
        # 重置所有相关变量
        search_model = None
        search_index = None
        search_image_map = None
        SEARCH_COMPONENTS_LOADED = False

except Exception as e:
    print(f"警告：语义搜索组件加载失败: {e}")
    search_model = None
    search_index = None
    search_image_map = None
    SEARCH_COMPONENTS_LOADED = False

# --- 应用程序初始化 ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# --- 全局常量定义 ---

# Nudity-NSFW 模型支持的所有标签类别
ALL_CLASSES = [
    "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED", "FEET_EXPOSED", "BELLY_COVERED", "FEET_COVERED",
    "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE", "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED"
]
# 标签在前端的中文显示名
LABEL_DISPLAY_NAMES = {
    "FEMALE_GENITALIA_COVERED": "女性生殖器遮挡",
    "FACE_FEMALE": "女性面部",
    "BUTTOCKS_EXPOSED": "臀部暴露",
    "FEMALE_BREAST_EXPOSED": "女性胸部暴露",
    "FEMALE_GENITALIA_EXPOSED": "女性生殖器暴露",
    "MALE_BREAST_EXPOSED": "男性胸部暴露",
    "ANUS_EXPOSED": "肛门暴露",
    "FEET_EXPOSED": "足部暴露",
    "BELLY_COVERED": "腹部遮挡",
    "FEET_COVERED": "足部遮挡",
    "ARMPITS_COVERED": "腋下遮挡",
    "ARMPITS_EXPOSED": "腋下暴露",
    "FACE_MALE": "男性面部",
    "BELLY_EXPOSED": "腹部暴露",
    "MALE_GENITALIA_EXPOSED": "男性生殖器暴露",
    "ANUS_COVERED": "肛门遮挡",
    "FEMALE_BREAST_COVERED": "女性胸部遮挡",
    "BUTTOCKS_COVERED": "臀部遮挡"
}
# 存放图片和JSON标注文件的主目录
IMAGE_DIR = 'images'
# 主目录下不进行JSON文件关联的子文件夹列表
SUB_FOLDERS_NO_JSON = ["no_detection"]
# 前端下拉菜单中代表主目录的显示名称
DEFAULT_FOLDER_NAME = 'Default (with JSON)'
# 聚合所有可供前端选择的文件夹选项
ALL_FOLDER_OPTIONS = [DEFAULT_FOLDER_NAME] + SUB_FOLDERS_NO_JSON

# 记录已删除文件信息的日志文件路径
DELETED_LIST_FILE = 'deleted_list.txt'
# 回收站目录配置
RECYCLE_BIN_DIR = 'recycle_bin'
# 记录已删除文件信息的日志文件路径
DELETED_LIST_FILE = 'deleted_list.txt'
STICKER_DIR = 'sticker_uploads'

# --- 短期图片缓存 ---
# 使用一个固定大小的双端队列来存储最近返回过的图片路径，以避免在短期内重复出现同一张图片。
# 当队列满时，新的元素会自动将最旧的元素挤出。
RECENTLY_SHOWN_CACHE_SIZE = 50
recently_shown_images = deque(maxlen=RECENTLY_SHOWN_CACHE_SIZE)


# --- JSON 数据缓存 ---
class JsonCache:
    """
    一个用于预加载、缓存和查询 JSON 标注文件的类。

    设计目标:
    通过在应用启动时将所有 JSON 文件一次性读入内存，显著加快后续处理请求中
    对标注数据的访问速度，避免了每次请求都进行磁盘 I/O。
    使用线程池并发加载，以提高启动效率。
    """
    def __init__(self, image_dir: str):
        """
        初始化缓存。

        Args:
            image_dir (str): 存放图片和 JSON 的根目录。
        """
        self.image_dir = os.path.abspath(image_dir)
        # 核心缓存结构：以不带扩展名的绝对路径为键，存储解析后的 JSON 内容。
        self._by_image_stem: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]] = {}
        self._json_by_path: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]] = {}
        self._errors: List[Tuple[str, str]] = []
        self._loaded = False

    def _iter_json_files(self) -> List[str]:
        """
        遍历并收集所有需要被缓存的 JSON 文件路径。
        会跳过在 `SUB_FOLDERS_NO_JSON` 中定义的目录。
        """
        files = []
        for root, dirs, fnames in os.walk(self.image_dir):
            rel_path = os.path.relpath(root, self.image_dir)
            # 如果当前目录是需要跳过的目录，则直接忽略
            if rel_path != "." and rel_path.split(os.sep)[0] in SUB_FOLDERS_NO_JSON:
                continue
            for f in fnames:
                if f.lower().endswith(".json"):
                    files.append(os.path.join(root, f))
        return files

    def _load_one(self, json_path: str) -> Optional[Tuple[str, Union[Dict[str, Any], List[Dict[str, Any]]]]]:
        """
        加载单个 JSON 文件，并处理可能发生的异常。
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
            # 使用不带扩展名的绝对路径作为标准化键
            stem_abs = os.path.splitext(os.path.abspath(json_path))[0]
            return stem_abs, annotations
        except Exception as e:
            self._errors.append((json_path, str(e)))
            return None

    def load(self, max_workers: int = min(8, (os.cpu_count() or 4) * 2)) -> None:
        """
        使用线程池并发加载所有找到的 JSON 文件。
        """
        if not os.path.isdir(self.image_dir):
            self._loaded = True
            return
        
        json_files = self._iter_json_files()
        if not json_files:
            self._loaded = True
            return

        self._by_image_stem.clear()
        self._json_by_path.clear()
        self._errors.clear()

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(self._load_one, p) for p in json_files]
            for fut in as_completed(futures):
                item = fut.result()
                if item:
                    stem_abs, anns = item
                    self._by_image_stem[stem_abs] = anns
                    self._json_by_path[os.path.abspath(stem_abs + ".json")] = anns
        self._loaded = True

    def get_annotations_for_image(self, image_path: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        根据图片路径从缓存中获取其对应的标注数据。
        """
        if not self._loaded:
            self.load()
        stem_abs = os.path.splitext(os.path.abspath(image_path))[0]
        return self._by_image_stem.get(stem_abs, [])

    def invalidate_on_delete(self, image_path: str):
        """
        当一个文件被删除时，从缓存中移除其对应的条目，保持数据同步。
        """
        stem_abs = os.path.splitext(os.path.abspath(image_path))[0]
        self._by_image_stem.pop(stem_abs, None)
        self._json_by_path.pop(os.path.abspath(stem_abs + ".json"), None)

# 全局 JSON 缓存实例
JSON_CACHE = JsonCache(IMAGE_DIR)


# --- 语义搜索函数 ---

@torch.no_grad()
def encode_text(query: str) -> np.ndarray:
    """ ... (函数内容保持不变) ... """
    if not SEARCH_COMPONENTS_LOADED or search_model is None:
        raise RuntimeError("语义搜索组件未加载")
    tokens = tokenize([query]).to(device)
    txt_feat = search_model.encode_text(tokens)
    txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-12)
    return txt_feat.detach().cpu().numpy().astype("float32")

def search(query: str, top_k: int) -> List[Tuple[str, float]]:
    """
    基于文本查询搜索最相似的图片。
    (已更新以适配基于ID的索引)
    """
    # 修改: 检查 search_image_map 而不是 search_paths
    if not SEARCH_COMPONENTS_LOADED or search_index is None or search_image_map is None:
        return []

    if not query or not query.strip():
        return []

    if not hasattr(search_index, 'ntotal') or not hasattr(search_index, 'search'):
        print("警告：索引对象无效，搜索功能被禁用")
        return []

    if search_index.ntotal == 0:
        return []

    top_k = int(max(1, min(top_k, search_index.ntotal)))
    vec = encode_text(query)
    D, I = search_index.search(vec, top_k)
    
    # 修改: I[0] 现在是自定义的ID列表，而不是位置索引
    ids = I[0].tolist()
    scores = D[0].tolist()
    
    paired = []
    for img_id, sc in zip(ids, scores):
        # 通过ID从 image_map 中查找路径
        path = search_image_map.get(str(img_id))
        if path and os.path.isfile(path): # 确保路径存在且文件有效
            paired.append((path, float(sc)))
            
    return paired

def filter_search_results(search_results: List[Tuple[str, float]], excluded_labels: set, required_labels: set, random_labels: set, nsfw_filter: str) -> List[Dict[str, Any]]:
    """
    对搜索结果应用标签过滤。

    Args:
        search_results: 搜索结果列表 [(path, score), ...]
        excluded_labels: 排除标签集合
        required_labels: 必选标签集合
        random_labels: 随机标签集合
        nsfw_filter: NSFW过滤 ('all', 'nsfw', 'normal')

    Returns:
        过滤后的结果列表，每个元素包含 path, score, folder_name, nsfw_status, labels
    """
    filtered = []
    image_dir_abs = os.path.abspath(IMAGE_DIR)

    for path, score in search_results:
        # 确保路径在 images 目录内
        if not os.path.abspath(path).startswith(image_dir_abs):
            continue

        # 确定文件夹名称
        rel_path = os.path.relpath(path, IMAGE_DIR)
        if rel_path.startswith('no_detection'):
            folder_name = 'no_detection'
        else:
            folder_name = DEFAULT_FOLDER_NAME

        # 获取 JSON 数据
        json_data = JSON_CACHE.get_annotations_for_image(path)

        if isinstance(json_data, dict):
            nsfw_status = json_data.get('nsfw', 'normal')
            annotations = json_data.get('annotations', [])
        else:
            nsfw_status = 'normal'
            annotations = json_data if isinstance(json_data, list) else []

        # NSFW 过滤
        if nsfw_filter == 'nsfw' and nsfw_status.upper() != 'NSFW':
            continue
        if nsfw_filter == 'normal' and nsfw_status.upper() != 'NORMAL':
            continue

        # 获取标签
        image_labels = {ann.get('label') for ann in annotations if isinstance(ann, dict) and ann.get('label')}

        # 规则 1: 排除 (最高优先级)
        if excluded_labels and not image_labels.isdisjoint(excluded_labels):
            continue
        # 规则 2: 必选 (第二优先级)
        if not required_labels.issubset(image_labels):
            continue
        # 规则 3: 随机 (第三优先级)
        if random_labels and image_labels.isdisjoint(random_labels):
            continue

        # 添加到结果
        filtered.append({
            'path': path,
            'score': score,
            'folder_name': folder_name,
            'nsfw_status': nsfw_status,
            'labels': list(image_labels)
        })

    return filtered

# --- 图像效果生成函数 ---

def create_mosaic_effect(img: np.ndarray, block_size: int = 12, **kwargs) -> np.ndarray:
    """
    为图像区域生成马赛克效果。
    通过两次缩放（先缩小后放大）来实现，性能较高。
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    # 块大小越大，缩小比例越大
    bs = max(2, int(block_size))
    ds_w = max(1, w // bs)
    ds_h = max(1, h // bs)
    small = cv2.resize(img, (ds_w, ds_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def create_black_effect(img: np.ndarray, **kwargs) -> np.ndarray:
    """生成一个与输入图像同样大小的纯黑色图像。"""
    return np.zeros_like(img)

def create_blur_effect(img: np.ndarray, ksize: int = 51, **kwargs) -> np.ndarray:
    """为图像生成高斯模糊效果。核大小必须是奇数。"""
    ksize = ksize if ksize % 2 != 0 else ksize + 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def create_face_strip_effect(img: np.ndarray, box: List[int], **kwargs) -> np.ndarray:
    """
    在人脸区域的眼睛位置生成一个黑色条带效果（类似罪犯条）。
    """
    x, y, w, h = map(int, box)
    # 计算条带的垂直中心和高度，使其更贴近眼睛区域
    strip_y_center = y + int(h * 0.3)
    strip_height = int(h * 0.4)
    strip_y_start = max(0, strip_y_center - strip_height // 2)
    strip_y_end = min(img.shape[0], strip_y_center + strip_height // 2)
    x1, x2 = max(0, x), min(img.shape[1], x + w)
    
    if x1 < x2 and strip_y_start < strip_y_end:
        # 原地在图像上绘制黑色矩形，性能优于创建新图
        cv2.rectangle(img, (x1, strip_y_start), (x2, strip_y_end), (0, 0, 0), -1)
    return img

def create_no_effect(img: np.ndarray, **kwargs) -> np.ndarray:
    """一个占位符函数，不执行任何操作，直接返回原图。"""
    return img

def create_sticker_effect(img: np.ndarray, **kwargs) -> np.ndarray:
    """占位函数：贴纸效果由自定义逻辑处理。"""
    return img

def create_sharp_sketch_effect(img: np.ndarray, line_thickness: int = 1, detail_level: float = 0.6, **kwargs) -> np.ndarray:
    """
    锐利素描效果 - 主轮廓更清晰突出的黑底白线。
    
    参数:
    - line_thickness: 线条粗细 (1-5)
    - detail_level: 细节保留程度 (0.1-1.0，越大细节越多)
    """
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 增强对比度，让轮廓更明显
    enhanced = cv2.convertScaleAbs(gray, alpha=1.3, beta=10)
    
    # 轻微降噪但保留边缘
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 多层次边缘检测
    # 主轮廓 - 高阈值，捕获最重要的边缘
    main_edges = cv2.Canny(denoised, 80, 160)
    
    # 细节边缘 - 根据detail_level调整
    detail_threshold = int(50 * detail_level)
    detail_edges = cv2.Canny(denoised, detail_threshold, detail_threshold * 2)
    
    # 合并边缘，主轮廓权重更高
    edges_combined = cv2.bitwise_or(main_edges, detail_edges)
    
    # 增强主轮廓 - 膨胀操作让线条更粗更清晰
    if line_thickness > 1:
        kernel_size = min(5, max(1, line_thickness))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edges_combined = cv2.dilate(edges_combined, kernel, iterations=1)
    
    # 连接断裂的线条
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_combined = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, connect_kernel)
    
    # 创建高对比度的黑底白线效果
    sketch = np.zeros_like(img)
    sketch[edges_combined == 255] = [255, 255, 255]
    
    # 可选：轻微模糊线条边缘，让效果更自然
    sketch = cv2.GaussianBlur(sketch, (3, 3), 0.5)
    
    return sketch

# --- 效果应用辅助函数 ---

def apply_circular_mask(target_img: np.ndarray, effect_source_img: np.ndarray, box: List[int]) -> np.ndarray:
    """
    将效果以圆形区域应用到目标图像上。

    Args:
        target_img (np.ndarray): 将要被修改的目标图像。
        effect_source_img (np.ndarray): 包含了效果的源图像。
        box (list): 标注框 [x, y, w, h]，用于定义圆形的位置和大小。

    Returns:
        np.ndarray: 应用了圆形效果的图像。
    """
    x, y, w, h = map(int, box)
    H, W = target_img.shape[:2]
    center_x, center_y = x + w // 2, y + h // 2
    # 使用外切圆半径，确保完全覆盖矩形框
    radius = int(np.sqrt(w**2 + h**2) / 2)

    # 仅在圆的外接矩形区域内进行操作，避免处理整张大图，提升性能
    x1, y1 = max(0, center_x - radius), max(0, center_y - radius)
    x2, y2 = min(W, center_x + radius), min(H, center_y + radius)
    if x1 >= x2 or y1 >= y2:
        return target_img

    # 创建一个局部的圆形掩码
    local_h, local_w = y2 - y1, x2 - x1
    mask_local = np.zeros((local_h, local_w), dtype=np.uint8)
    cv2.circle(mask_local, (center_x - x1, center_y - y1), radius, (255,), -1)

    # 从源和目标图像中提取出待处理的局部区域 (ROI)
    src_roi = effect_source_img[y1:y2, x1:x2]
    dst_roi = target_img[y1:y2, x1:x2]
    
    # 使用掩码将源ROI的内容拷贝到目标ROI，实现圆形效果合成
    target_img[y1:y2, x1:x2] = np.where(mask_local[:, :, np.newaxis] > 0, src_roi, dst_roi)
    
    return target_img

# 将效果名称映射到其处理函数，便于在 API 中动态调用
EFFECT_CREATORS = {
    "none": create_no_effect,
    "mosaic": create_mosaic_effect,
    "black": create_black_effect,
    "blur": create_blur_effect,
    "face_strip": create_face_strip_effect,
    "sharp_sketch": create_sharp_sketch_effect,
    "sticker": create_sticker_effect,
}

# --- 贴纸遮挡辅助函数 ---

def apply_sticker_overlay(target_img: np.ndarray, sticker_img: np.ndarray, box: List[int]) -> np.ndarray:
    """
    将贴纸覆盖到目标图像的指定区域。
    支持带 alpha 通道的 PNG。
    """
    x, y, w, h = map(int, box)
    if w <= 0 or h <= 0:
        return target_img

    H, W = target_img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x1 >= x2 or y1 >= y2:
        return target_img

    sticker_resized = cv2.resize(sticker_img, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
    roi = target_img[y1:y2, x1:x2]

    if sticker_resized.shape[2] == 4:
        sticker_rgb = sticker_resized[:, :, :3].astype(np.float32)
        alpha = sticker_resized[:, :, 3].astype(np.float32) / 255.0
        alpha = alpha[:, :, np.newaxis]
        blended = (sticker_rgb * alpha + roi.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        target_img[y1:y2, x1:x2] = blended
    else:
        target_img[y1:y2, x1:x2] = sticker_resized[:, :, :3]

    return target_img


# --- Flask 路由定义 ---

@app.route('/')
def index():
    """
    渲染应用程序的主页面 (index.html)。
    将所有可用的类别、处理方法和文件夹选项传递给模板，用于动态生成前端UI。
    """
    return render_template(
        'index.html',
        all_classes=ALL_CLASSES,
        label_display=LABEL_DISPLAY_NAMES,
        methods=list(EFFECT_CREATORS.keys()),
        folders=ALL_FOLDER_OPTIONS
    )

@app.route('/process', methods=['POST'])
def process_image():
    """
    核心图像处理 API 端点。

    工作流程:
    1.  解析前端发送的 JSON 请求，获取用户选择的各项参数。
    2.  根据选择的文件夹，递归搜集所有符合条件的图片路径。
    3.  应用一个三级优先级的筛选逻辑来过滤图片：
        -   最高优先级 (排除): 任何包含“排除标签”的图片都将被过滤掉。
        -   第二优先级 (必选): 图片必须包含所有“必选标签”。
        -   第三优先级 (随机): 如果设置了，图片必须至少包含一个“随机标签”。
    4.  从最终候选列表中随机选择一张图片（利用短期缓存避免重复）。
    5.  读取图片，并根据标注信息和用户设置，应用相应的视觉效果。
    6.  将处理后的图片编码为 Base64 字符串，连同元数据一起返回给前端。
    """
    try:
        # 1. 解析前端请求
        data = request.get_json()
        selections = data.get('selections', {})
        excluded_labels = set(data.get('excluded_labels', []))
        required_labels = set(data.get('required_labels', []))
        random_labels = set(selections.keys())
        folder_choices = data.get('folders', [])
        all_methods = data.get('all_methods', {})
        apply_circle = data.get('apply_circle', False)
        nsfw_filter = data.get('nsfw_filter', 'all')
        search_query = data.get('search_query')
        search_results_from_req = data.get('search_results', [])
        current_search_index = data.get('current_search_index', 0)

        # 新的逻辑分支
        selected_image_info = None

        # 分支 1: 用户正在浏览已有的搜索结果
        if search_results_from_req:
            if not search_results_from_req or current_search_index >= len(search_results_from_req):
                return jsonify({"error": "搜索结果为空或索引超出范围"}), 400
            selected_result = search_results_from_req[current_search_index]
            selected_image_info = {"path": selected_result['path'], "folder_name": selected_result.get('folder_name', DEFAULT_FOLDER_NAME)}
            recently_shown_images.append(selected_image_info['path'])
        
        # 分支 2: 用户发起了新的搜索请求 (此逻辑被合并到下面)
        # 分支 3: 用户没有使用搜索，而是通过标签和文件夹随机选择
        else:
            images_with_json_potential = []
            images_without_json = []

            # 如果有搜索查询，则覆盖文件列表
            if search_query and SEARCH_COMPONENTS_LOADED:
                # 注意：这里我们只处理带JSON的图片，因为搜索结果需要标签过滤
                search_hits = search(search_query, 100) # 假设最多返回100个结果
                for path, score in search_hits:
                    if os.path.splitext(path)[0] + '.json':
                        rel_path = os.path.relpath(path, IMAGE_DIR)
                        folder_name = 'no_detection' if rel_path.startswith('no_detection') else DEFAULT_FOLDER_NAME
                        # 确保搜索结果在用户选择的文件夹范围内
                        if folder_name in folder_choices:
                             images_with_json_potential.append({"path": path, "folder_name": folder_name})
            
            # 如果没有搜索，则从磁盘加载
            else:
                for choice in folder_choices:
                    target_dir = ""
                    is_no_json_folder = choice in SUB_FOLDERS_NO_JSON

                    if choice == DEFAULT_FOLDER_NAME:
                        target_dir = IMAGE_DIR
                    elif is_no_json_folder:
                        target_dir = os.path.join(IMAGE_DIR, choice)

                    if not (target_dir and os.path.isdir(target_dir)):
                        continue

                    for root, dirs, files in os.walk(target_dir):
                        # 如果是主目录，则确保不进入 `no_json` 子目录
                        if choice == DEFAULT_FOLDER_NAME:
                            dirs[:] = [d for d in dirs if d not in SUB_FOLDERS_NO_JSON]

                        for f in files:
                            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                                img_path = os.path.join(root, f)
                                img_info = {"path": img_path, "folder_name": choice}
                                json_path = os.path.splitext(img_path)[0] + '.json'

                                if is_no_json_folder or not os.path.exists(json_path):
                                    images_without_json.append(img_info)
                                else:
                                    images_with_json_potential.append(img_info)

                        if is_no_json_folder:
                            break # no_json 文件夹不递归

            # 3. 应用标签筛选逻辑
            candidate_images_with_json = []
            for img_info in images_with_json_potential:
                json_data = JSON_CACHE.get_annotations_for_image(img_info["path"])

                if isinstance(json_data, dict):
                    nsfw_status = json_data.get('nsfw', 'normal')
                    annotations = json_data.get('annotations', [])
                else:
                    nsfw_status = 'normal'
                    annotations = json_data if isinstance(json_data, list) else []

                # NSFW 过滤
                if nsfw_filter == 'nsfw' and nsfw_status.upper() != 'NSFW':
                    continue
                if nsfw_filter == 'normal' and nsfw_status.upper() != 'NORMAL':
                    continue

                image_labels = {ann.get('label') for ann in annotations if isinstance(ann, dict) and ann.get('label')}

                # 规则 1: 排除 (最高优先级)
                if excluded_labels and not image_labels.isdisjoint(excluded_labels):
                    continue
                # 规则 2: 必选 (第二优先级)
                if not required_labels.issubset(image_labels):
                    continue
                # 规则 3: 随机 (第三优先级)
                if random_labels and image_labels.isdisjoint(random_labels):
                    continue

                candidate_images_with_json.append(img_info)

            # 4. 合并候选列表并选择图片
            # (无JSON的图片不受标签筛选影响，直接加入最终列表)
            images_to_choose_from = candidate_images_with_json + images_without_json

            if not images_to_choose_from:
                error_msg = "在所选文件夹中没有找到任何符合条件的图片。"
                return jsonify({"error": error_msg}), 404

            # 从未在短期内展示过的图片中选择，如果都展示过了，则重置选择池
            available_choices = [img for img in images_to_choose_from if img['path'] not in recently_shown_images]
            if not available_choices:
                available_choices = images_to_choose_from

            selected_image_info = random.choice(available_choices)
            recently_shown_images.append(selected_image_info['path']) # 更新短期缓存

        # 5. 读取并处理图片
        img_path = selected_image_info["path"]
        original_img = cv2.imread(img_path)
        if original_img is None:
            return jsonify({"error": f"无法读取图片: {img_path}"}), 500

        # 统一缩放预览图，避免前端显示过大图片
        max_side, scale = 1920, 1.0
        oh, ow = original_img.shape[:2]
        if max(oh, ow) > max_side:
            scale = max_side / float(max(oh, ow))
            new_w, new_h = int(round(ow * scale)), int(round(oh * scale))
            original_img = cv2.resize(original_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        processed_img = original_img.copy()
        json_data = JSON_CACHE.get_annotations_for_image(img_path)

        if isinstance(json_data, dict):
            annotations = json_data.get('annotations', [])
        else:
            annotations = json_data if isinstance(json_data, list) else []

        # 如果图片被缩放，则同步缩放标注框的坐标
        if scale != 1.0 and annotations:
            for ann in annotations:
                if 'box' in ann:
                    ann['box'] = [int(round(c * scale)) for c in ann['box']]

        # 分离正常效果和反转效果的任务
        normal_tasks, reveal_tasks = [], []
        for ann in annotations:
            label = ann.get('label')
            if label in all_methods and 'box' in ann:
                task = {'ann': ann, 'details': all_methods[label]}
                if all_methods[label].get('invert', False):
                    reveal_tasks.append(task)
                else:
                    normal_tasks.append(task)
        
        # 优先处理反选（"reveal"）效果
        if reveal_tasks:
            # 创建一个掩码，标记所有需要“揭示”的区域
            reveal_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
            for task in reveal_tasks:
                box = task['ann']['box']
                x, y, w, h = map(int, box)
                if apply_circle:
                    cv2.circle(reveal_mask, (x + w // 2, y + h // 2), int(np.sqrt(w**2 + h**2) / 2), (255,), -1)
                else:
                    cv2.rectangle(reveal_mask, (x, y), (x + w, y + h), 255, -1)
            
            # 使用最后一个反选任务定义背景效果
            last_reveal_task = reveal_tasks[-1]
            creator_func = EFFECT_CREATORS.get(last_reveal_task['details']['method'], create_no_effect)
            background_effect_img = creator_func(original_img.copy(), box=last_reveal_task['ann']['box'])
            # 将原图内容（揭示区）和背景效果（非揭示区）合并
            processed_img = np.where(reveal_mask[:, :, np.newaxis] > 0, original_img, background_effect_img)

        # 处理正常效果
        for task in normal_tasks:
            box = task['ann']['box']
            x, y, w, h = map(int, box)
            if w <= 0 or h <= 0:
                continue

            method = task['details'].get('method')
            if method == 'sticker':
                sticker_path = task['details'].get('sticker_path')
                if sticker_path:
                    sticker_abs = os.path.abspath(sticker_path)
                    sticker_root = os.path.abspath(STICKER_DIR)
                    if sticker_abs.startswith(sticker_root) and os.path.isfile(sticker_abs):
                        sticker_img = cv2.imread(sticker_abs, cv2.IMREAD_UNCHANGED)
                        if sticker_img is not None and sticker_img.ndim == 3:
                            processed_img = apply_sticker_overlay(processed_img, sticker_img, box)
                continue

            creator_func = EFFECT_CREATORS.get(method, create_no_effect)
            roi_src = processed_img[y:y+h, x:x+w]
            
            if method == 'face_strip':
                # face_strip 需要在更大范围上操作，直接传递整个处理图
                processed_img = creator_func(processed_img, box=box)
            elif apply_circle:
                # 为圆形效果创建临时源
                effect_source_full = processed_img.copy()
                effect_source_full[y:y+h, x:x+w] = creator_func(roi_src, box=box)
                processed_img = apply_circular_mask(processed_img, effect_source_full, box)
            else:
                # 直接在矩形 ROI 内应用效果
                processed_img[y:y+h, x:x+w] = creator_func(roi_src, box=box)

        # 6. 编码并返回结果
        _, buffer_proc = cv2.imencode('.jpg', processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        b64_processed = base64.b64encode(buffer_proc.tobytes()).decode('utf-8')

        return jsonify({
            'filename': f"{selected_image_info['folder_name']} / {os.path.basename(img_path)}",
            'processed_image': b64_processed,
            'image_path': img_path
        })

    except Exception as e:
        app.logger.error(f"处理图片时发生严重错误: {e}", exc_info=True)
        return jsonify({"error": "服务器内部错误，请联系管理员。"}), 500

@app.route('/delete', methods=['POST'])
def delete_file():
    """
    API 端点，用于安全地将图片及其关联文件移动到回收站，并更新索引。
    (已完全重写以适配新架构)
    """
    try:
        if not request.json:
            return jsonify({"error": "请求体必须是JSON格式。"}), 400
        path_to_delete = request.json.get('path')
        if not path_to_delete:
            return jsonify({"error": "请求中未提供文件路径。"}), 400

        # --- 安全性校验 ---
        image_storage_dir = os.path.abspath(IMAGE_DIR)
        requested_path = os.path.abspath(path_to_delete)
        if not requested_path.startswith(image_storage_dir):
            app.logger.warning(f"检测到不安全的删除请求，已拒绝: {requested_path}")
            return jsonify({"error": "禁止访问：无权删除此路径下的文件。"}), 403

        # --- 1. 将文件移动到回收站 ---
        image_path = requested_path
        json_path = os.path.splitext(image_path)[0] + '.json'
        moved_files_basenames = []

        # 为移动的文件生成带时间戳的新名称，防止冲突
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if os.path.exists(image_path):
            base_name = os.path.basename(image_path)
            dest_path = os.path.join(RECYCLE_BIN_DIR, f"{timestamp}_{base_name}")
            shutil.move(image_path, dest_path)
            moved_files_basenames.append(base_name)

        if os.path.exists(json_path):
            base_name = os.path.basename(json_path)
            dest_path = os.path.join(RECYCLE_BIN_DIR, f"{timestamp}_{base_name}")
            shutil.move(json_path, dest_path)
            moved_files_basenames.append(base_name)
        
        # 从内存缓存中移除
        JSON_CACHE.invalidate_on_delete(image_path)

        if not moved_files_basenames:
             return jsonify({"error": "文件未找到，可能已被其他操作删除。"}), 404

        # --- 2. 从索引和元数据中移除条目 ---
        image_id_to_remove = None
        if SEARCH_COMPONENTS_LOADED and search_index is not None:
             # 加载最新的元数据进行操作
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # 创建反向映射以通过路径查找ID
            path_to_id = {v: k for k, v in metadata.get("image_map", {}).items()}
            
            if path_to_delete in path_to_id:
                image_id_to_remove_str = path_to_id[path_to_delete]
                image_id_to_remove_arr = np.array([int(image_id_to_remove_str)], dtype=np.int64)

                # a. 从Faiss索引中移除
                search_index.remove_ids(image_id_to_remove_arr)
                
                # b. 从元数据中移除
                del metadata["image_map"][image_id_to_remove_str]
                
                # c. 保存更新后的索引和元数据
                faiss.write_index(search_index, INDEX_FILE)
                with open(METADATA_FILE, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                # d. 更新全局变量以反映变化
                global search_image_map
                search_image_map = metadata.get("image_map", {})
                
                print(f"成功从索引中移除 ID: {image_id_to_remove_arr[0]} 对应的 '{path_to_delete}'")
                image_id_to_remove = image_id_to_remove_arr[0]
            else:
                print(f"警告: '{path_to_delete}' 已被移动，但在索引元数据中未找到，无需更新索引。")

        # --- 3. 记录操作日志 ---
        log_timestamp = datetime.datetime.now().isoformat()
        log_entry = f"{log_timestamp} - Moved to recycle bin: {', '.join(moved_files_basenames)}\n"
        try:
            with open(DELETED_LIST_FILE, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except IOError as io_err:
            app.logger.error(f"无法写入日志文件 {DELETED_LIST_FILE}: {io_err}")

        return jsonify({
            "success": True,
            "message": f"成功将文件移动到回收站: {', '.join(moved_files_basenames)}",
            "index_updated": image_id_to_remove is not None
        })

    except Exception as e:
        app.logger.error(f"删除文件时发生错误: {e}", exc_info=True)
        return jsonify({"error": "服务器在删除文件时发生内部错误。"}), 500

@app.route('/search', methods=['POST'])
def search_images():
    """
    语义搜索 API 端点。

    接受查询文本和标签过滤参数，返回过滤后的搜索结果列表。
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求体必须是JSON格式。"}), 400

        query = data.get('query')
        if not query or not query.strip():
            return jsonify({"error": "查询文本不能为空。"}), 400

        top_k = data.get('top_k', 100)
        excluded_labels = set(data.get('excluded_labels', []))
        required_labels = set(data.get('required_labels', []))
        random_labels = set(data.get('random_labels', []))
        nsfw_filter = data.get('nsfw_filter', 'all')

        # 获取搜索结果
        search_results = search(query, top_k)

        # 应用标签过滤
        filtered_results = filter_search_results(search_results, excluded_labels, required_labels, random_labels, nsfw_filter)

        return jsonify({"results": filtered_results})

    except Exception as e:
        app.logger.error(f"搜索时发生错误: {e}", exc_info=True)
        return jsonify({"error": "服务器内部错误。"}), 500

@app.route('/upload_sticker', methods=['POST'])
def upload_sticker():
    """
    上传贴纸图片并返回服务器保存路径。
    """
    try:
        if 'sticker' not in request.files:
            return jsonify({"error": "未找到上传文件。"}), 400
        file = request.files['sticker']
        label = request.form.get('label', 'unknown')
        if not file or file.filename == '':
            return jsonify({"error": "文件名为空。"}), 400

        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ['.png', '.jpg', '.jpeg']:
            return jsonify({"error": "仅支持 PNG/JPG/JPEG 贴纸。"}), 400

        os.makedirs(STICKER_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_label = "".join([c for c in label if c.isalnum() or c in ['_', '-']]) or "sticker"
        saved_name = f"{safe_label}_{timestamp}{ext}"
        save_path = os.path.join(STICKER_DIR, saved_name)
        file.save(save_path)

        return jsonify({"sticker_path": os.path.abspath(save_path)})
    except Exception as e:
        app.logger.error(f"上传贴纸时发生错误: {e}", exc_info=True)
        return jsonify({"error": "服务器内部错误。"}), 500

# --- 应用程序入口点 ---
if __name__ == '__main__':
    # 确保所有必要的目录在启动时都存在
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(RECYCLE_BIN_DIR, exist_ok=True)
    os.makedirs(STICKER_DIR, exist_ok=True)
    for sub_folder in SUB_FOLDERS_NO_JSON:
        path = os.path.join(IMAGE_DIR, sub_folder)
        os.makedirs(path, exist_ok=True)

    # 应用启动时，预先加载所有 JSON 数据到缓存中
    print("Pre-loading JSON annotations into cache...")
    JSON_CACHE.load()
    print("JSON cache loaded.")

    # 以生产模式启动服务器
    app.run(debug=False, host='0.0.0.0', port=5001)

