import os
import re

def sanitize_filename(filename):
    """替换文件名中的空格和非法字符为下划线"""
    # 定义需要替换的非法字符（可以根据需求增减）
    illegal_chars = r'[<>:"/\\|?*\s]'
    
    # 将非法字符和空格替换为下划线
    new_name = re.sub(illegal_chars, '_', filename)
    
    # 去除连续的下划线
    new_name = re.sub(r'_+', '_', new_name).strip('_')
    
    return new_name

def rename_files(start_dir):
    """遍历目录并重命名文件"""
    for root, dirs, files in os.walk(start_dir):
        for filename in files:
            # 获取原始文件路径
            old_path = os.path.join(root, filename)
            
            # 生成新文件名
            new_filename = sanitize_filename(filename)
            new_path = os.path.join(root, new_filename)
            
            # 跳过不需要修改的文件名
            if new_filename == filename:
                continue
            
            # 处理文件名冲突（如果新文件名已存在）
            counter = 1
            temp_new_path = new_path
            while os.path.exists(temp_new_path):
                name_parts = os.path.splitext(new_filename)
                temp_new_path = os.path.join(
                    root, 
                    f"{name_parts[0]}_{counter}{name_parts[1]}"
                )
                counter += 1
            
            # 重命名文件
            os.rename(old_path, temp_new_path)
            print(f"Renamed: {old_path} -> {temp_new_path}")

if __name__ == "__main__":
    target_dir = input("请输入要处理的文件夹路径: ").strip()
    if os.path.isdir(target_dir):
        rename_files(target_dir)
        print("文件重命名完成！")
    else:
        print("错误：输入的路径不是有效的文件夹")
