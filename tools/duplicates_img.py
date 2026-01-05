import os
import hashlib
from collections import defaultdict
import shutil

# --- 用户配置 ---

# 1. 设置要扫描的图片文件夹路径
#    请确保使用正确的路径，例如 r'C:\Users\YourUser\Pictures'
IMAGE_DIR = r'C:\Users\YourUser\image_processor_app\images\no_detection'

# 2. 设置处理方式:
#    'list'  - 仅打印重复文件列表，不移动或删除
#    'move'  - 将重复文件移动到 DUPLICATE_DIR 指定的文件夹
#    'delete'- 直接删除重复文件（请谨慎使用！）
ACTION = 'delete'

# 3. 如果 ACTION 设置为 'move'，请指定用于存放重复文件的文件夹
DUPLICATE_DIR = os.path.join(IMAGE_DIR, 'duplicates_backup')


# --- 脚本正文 ---

def get_file_hash(filepath):
    """计算文件的 SHA256 哈希值"""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    except IOError as e:
        print(f"无法读取文件 {filepath}: {e}")
        return None

def find_duplicates(image_dir):
    """查找目录中内容完全相同的重复图片"""
    file_hashes = defaultdict(list)
    duplicates = set()

    print("第一阶段：扫描文件，计算哈希值...")
    for root, _, files in os.walk(image_dir):
        # 跳过备份文件夹本身
        if root == DUPLICATE_DIR:
            continue
        for filename in files:
            # 过滤常见图片格式
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                filepath = os.path.join(root, filename)
                
                # 计算文件哈希
                file_hash = get_file_hash(filepath)
                if file_hash:
                    file_hashes[file_hash].append(filepath)

    print("\n第二阶段：查找内容完全相同的重复文件...")
    for file_list in file_hashes.values():
        if len(file_list) > 1:
            # 按文件路径字母顺序排序，默认保留第一个文件
            file_list.sort()
            # 将除第一个文件外的所有其他文件标记为重复项
            for item in file_list[1:]:
                duplicates.add(item)
                print(f"  [精确重复] 发现: {item} (将保留: {file_list[0]})")
    
    return list(duplicates)

def process_duplicates(duplicate_images, action):
    """
    根据指定的 action 处理重复文件。
    新增逻辑：同时处理与图片同名的 .json 文件。
    """
    if not duplicate_images:
        print("\n未发现任何重复文件。")
        return

    # --- 新增逻辑：查找关联的 .json 文件 ---
    all_files_to_process = []
    print("\n第三阶段：检查关联的 .json 文件...")
    for img_path in duplicate_images:
        all_files_to_process.append(img_path)
        # 构造同名的 .json 文件路径
        base_name, _ = os.path.splitext(img_path)
        json_path = base_name + '.json'
        
        # 检查 .json 文件是否存在
        if os.path.exists(json_path):
            all_files_to_process.append(json_path)
            print(f"  [关联文件] 发现: {json_path}")

    print(f"\n总共发现 {len(duplicate_images)} 个重复图片及其关联文件，共计 {len(all_files_to_process)} 个文件。")
    print(f"准备执行操作: {action}")

    if action == 'list':
        print("\n待处理文件列表:")
        for f in all_files_to_process:
            print(f)
    
    elif action == 'move':
        if not os.path.exists(DUPLICATE_DIR):
            os.makedirs(DUPLICATE_DIR)
            print(f"已创建备份文件夹: {DUPLICATE_DIR}")
        
        for f in all_files_to_process:
            try:
                # 确保目标文件名不冲突
                base_name = os.path.basename(f)
                dest_path = os.path.join(DUPLICATE_DIR, base_name)
                # 如果文件名已存在，添加唯一后缀
                count = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(base_name)
                    dest_path = os.path.join(DUPLICATE_DIR, f"{name}_{count}{ext}")
                    count += 1
                
                shutil.move(f, dest_path)
                print(f"已移动: {f} -> {dest_path}")
            except Exception as e:
                print(f"移动文件失败: {f}, 错误: {e}")

    elif action == 'delete':
        # 在删除前给予用户最后确认机会
        confirm = input(f"警告：您将永久删除 {len(all_files_to_process)} 个文件（包括图片和JSON）。是否继续？(输入 'yes' 确认): ")
        if confirm.lower() == 'yes':
            for f in all_files_to_process:
                try:
                    os.remove(f)
                    print(f"已删除: {f}")
                except Exception as e:
                    print(f"删除文件失败: {f}, 错误: {e}")
        else:
            print("操作已取消。")
            
    else:
        print(f"错误：未知的操作 '{action}'。请选择 'list', 'move', 或 'delete'。")


if __name__ == "__main__":
    if not os.path.isdir(IMAGE_DIR):
        print(f"错误：指定的文件夹不存在 -> {IMAGE_DIR}")
    else:
        duplicate_list = find_duplicates(IMAGE_DIR)
        process_duplicates(duplicate_list, ACTION)
        print("\n处理完成。")
