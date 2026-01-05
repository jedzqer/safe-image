# index_img.py - 增量同步图片索引（支持增、删），使用稳定 ID
import os, glob, json, numpy as np
import torch, faiss
from torch.cuda.amp import autocast
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from cn_clip.clip import load_from_name

print("开始图片索引同步...")

# --- 配置 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
index_file = "image_features.index"
metadata_file = "index_metadata.json" # 新的元数据文件
img_dir = "images"
model_name = "ViT-L-14"
batch_size_init = 36 # 初始 batch size

# --- 环境设置 ---
torch.set_num_threads(os.cpu_count() or 8)
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 8)
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 8)

# --- 1) 加载模型 ---
print("正在加载 CLIP 模型...")
model, preprocess = load_from_name(model_name, device=device)
model.eval()

# --- 2) 加载现有索引和元数据 ---
index = None
metadata = {"next_id": 0, "image_map": {}}
path_to_id = {}
valid_new_paths = []

if os.path.exists(index_file) and os.path.exists(metadata_file):
    print("正在加载现有索引和元数据...")
    try:
        index = faiss.read_index(index_file)
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        # 创建一个反向映射，方便通过路径查找ID
        path_to_id = {v: k for k, v in metadata["image_map"].items()}
        print(f"加载完成。当前索引中有 {index.ntotal} 张图片。")
    except Exception as e:
        print(f"警告：加载索引文件失败，将重新创建。错误: {e}")
        index = None
        metadata = {"next_id": 0, "image_map": {}}

# --- 3) 扫描磁盘文件，确定变更 ---
print("正在扫描图片目录以确定变更...")
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
disk_paths = set(p for p in glob.glob(os.path.join(img_dir, "**", "*"), recursive=True) if os.path.splitext(p)[1].lower() in exts)

if not disk_paths:
    print(f"错误：在 '{img_dir}' 中没有找到匹配的图片文件。")
    # 如果索引存在但文件夹空了，则清空索引
    if os.path.exists(index_file): os.remove(index_file)
    if os.path.exists(metadata_file): os.remove(metadata_file)
    print("已清空旧索引（如果存在）。")
    raise SystemExit(1)

indexed_paths = set(metadata["image_map"].values())

paths_to_add = sorted(list(disk_paths - indexed_paths))
paths_to_remove = sorted(list(indexed_paths - disk_paths))

if not paths_to_add and not paths_to_remove:
    print("索引与文件系统一致，无需更新。")
    raise SystemExit(0)

# --- 4) 处理被删除的图片 ---
if paths_to_remove:
    print(f"检测到 {len(paths_to_remove)} 张图片已被删除，正在从索引中移除...")
    
    ids_to_remove = [int(path_to_id[p]) for p in paths_to_remove if p in path_to_id]
    
    if ids_to_remove:
        if index is None:
            raise RuntimeError("索引为空，无法移除已删除图片。")
        index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
        # 更新元数据
        for p in paths_to_remove:
            if p in path_to_id:
                del metadata["image_map"][path_to_id[p]]
                del path_to_id[p]
        print(f"成功移除了 {len(ids_to_remove)} 个索引。")

# --- 5) 处理新增的图片 ---
if paths_to_add:
    print(f"发现 {len(paths_to_add)} 张新图片，开始处理...")

    new_features_list = []
    new_ids = []
    valid_new_paths = []
    
    # helper for processing a single image
    def encode_one(p):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                return preprocess(im)
        except (UnidentifiedImageError, OSError) as e:
            print(f"警告：跳过无法读取的文件: {p} ({e})")
            return None

    pbar = tqdm(total=len(paths_to_add), desc="提取新图片特征", unit="img")
    
    i = 0
    batch_size = batch_size_init
    while i < len(paths_to_add):
        end = min(i + batch_size, len(paths_to_add))
        batch_paths = paths_to_add[i:end]

        imgs, keep_idx = [], []
        for idx, p in enumerate(batch_paths):
            tensor = encode_one(p)
            if tensor is not None:
                imgs.append(tensor)
                keep_idx.append(idx)
            pbar.update(1)

        if not imgs:
            i = end
            continue

        imgs = torch.stack(imgs, dim=0).pin_memory()

        try:
            with torch.no_grad():
                with autocast(dtype=torch.float16) if device == "cuda" else torch.no_grad():
                    inputs = imgs.to(device, non_blocking=True)
                    feats = model.encode_image(inputs).float() # Keep as float32 for CPU
                    feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            
            # 收集有效的特征、路径和新ID
            new_features_list.append(feats.cpu().numpy())
            current_next_id = metadata["next_id"]
            for k_idx, p_idx in enumerate(keep_idx):
                path = batch_paths[p_idx]
                new_id = current_next_id + k_idx
                valid_new_paths.append(path)
                new_ids.append(new_id)

            metadata["next_id"] += len(keep_idx)
            i = end

        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and device == "cuda":
                torch.cuda.empty_cache()
                if batch_size > 1:
                    new_bs = max(1, batch_size // 2)
                    print(f"\n警告：检测到显存不足，batch_size {batch_size} -> {new_bs}，重试该批。")
                    batch_size = new_bs
                else:
                    print(f"\n错误：batch_size=1 仍 OOM，跳过当前图像: {paths_to_add[i]}")
                    i += 1
            else:
                raise
    pbar.close()

    if new_features_list:
        new_features = np.concatenate(new_features_list, axis=0)
        new_ids = np.array(new_ids, dtype=np.int64)
        print(f"成功提取 {len(valid_new_paths)} 张新图片的特征。")

        # 初始化索引（如果不存在）
        if index is None:
            print("首次创建索引...")
            d = new_features.shape[1]
            # 使用 IndexFlatIP 作为基础索引，然后用 IndexIDMap 包裹
            base_index = faiss.IndexFlatIP(d)
            index = faiss.IndexIDMap(base_index)

        # 添加新特征和ID
        if index is None:
            raise RuntimeError("索引为空，无法添加新图片特征。")
        index.add_with_ids(new_features, new_ids)  # type: ignore[call-arg]
        
        # 更新元数据
        for path, new_id in zip(valid_new_paths, new_ids):
            metadata["image_map"][str(new_id)] = path
    else:
        print("未成功提取任何新图片特征。")

# --- 6) 保存更新后的索引和元数据 ---
if paths_to_add or paths_to_remove:
    print(f"正在保存更新后的索引到 {index_file}...")
    faiss.write_index(index, index_file)

    print(f"正在保存更新后的元数据到 {metadata_file}...")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n索引同步完成！")
    if index is not None:
        print(f"总图片数量: {index.ntotal}")
    if paths_to_add: print(f"本次新增: {len(valid_new_paths) if 'valid_new_paths' in locals() else 0}")
    if paths_to_remove: print(f"本次移除: {len(paths_to_remove)}")
else:
    print("无变更，无需保存。")
