import os
import json
from PIL import Image
from transformers import pipeline # pyright: ignore[reportMissingImports]

# 设置图片目录和模型路径
image_dir = "./SafeNet"  # 请根据实际路径调整
model_path = "./NSFWdetector/"  # 请确认模型路径正确

# 初始化分类器
classifier = pipeline("image-classification", model=model_path)

# 遍历目录中的所有图片文件
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = os.path.join(image_dir, filename)
        json_path = os.path.splitext(image_path)[0] + '.json'
        
        # 检查是否存在对应的 JSON 文件
        if not os.path.exists(json_path):
            print(f"警告: 未找到 {filename} 对应的 JSON 文件，跳过处理。")
            continue
        
        # 打开图片并进行分类
        try:
            img = Image.open(image_path)
            results = classifier(img)
        except Exception as e:
            print(f"处理图片 {filename} 时出错: {e}")
            continue
        
        # 解析分类结果，确定 nsfw 标签
        nsfw_label = "normal"
        for result in results:
            if "nsfw" in result['label'].lower() and result['score'] > 0.5:
                nsfw_label = "NSFW"
                break
        
        # 读取现有 JSON 文件内容
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except Exception as e:
            print(f"读取 JSON 文件 {json_path} 时出错: {e}")
            continue
        
        # 构建新格式的 JSON 数据
        new_data = {
            "nsfw": nsfw_label,
            "annotations": existing_data  # 将原有数组移到 annotations 键下
        }
        
        # 写回 JSON 文件
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)
            print(f"已更新: {json_path} (NSFW: {nsfw_label})")
        except Exception as e:
            print(f"写入 JSON 文件 {json_path} 时出错: {e}")

print("批量处理完成！")
