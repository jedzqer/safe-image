import cv2
import os
import shutil
import json  # 导入 json 模块
import threading
from nudenet import NudeDetector

# 📂 文件夹路径
input_folder = './input'
output_folder = './SafeNet'
no_detection_folder = './no_detection'

# 🛠️ 创建输出目录
os.makedirs(output_folder, exist_ok=True)
os.makedirs(no_detection_folder, exist_ok=True)

# 线程工作函数
def worker(file_list, thread_id):
    """
    工作线程，负责检测图片并保存原始图片及元数据。
    """
    # 每个线程创建自己的检测器实例
    detector = NudeDetector(inference_resolution=640)
    
    for filename in file_list:
        input_path = os.path.join(input_folder, filename)
        
        # 1. 使用AI检测图像
        try:
            detections = detector.detect(input_path)
        except Exception as e:
            print(f"❌ 线程{thread_id} 处理 {filename} 时出错: {e}")
            continue

        # 2. 收集需要保存的元数据
        metadata_to_save = []

        for det in detections:
            class_name = det['class']
            
            if class_name:

                # 准备要保存的检测信息
                detection_info = {
                    'label': class_name,
                    'box': det['box']
                }
                metadata_to_save.append(detection_info)

        # 3. 根据检测结果处理文件
        if metadata_to_save:
            # 获取不带扩展名的文件名，用于生成输出文件
            base_filename = os.path.splitext(filename)[0]
            extension = os.path.splitext(filename)[1]

            # 定义输出路径
            output_image_path = os.path.join(output_folder, filename)
            output_json_path = os.path.join(output_folder, base_filename + '.json')

            # 移动原始图片
            shutil.move(input_path, output_image_path)

            # 保存元数据到JSON文件
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_to_save, f, indent=4, ensure_ascii=False)
            
            print(f"✅ 线程{thread_id} 检测到目标: {filename} -> 已保存图片和元数据")
        else:
            # 没有检测到目标，移动到 no_detection 文件夹
            new_path = os.path.join(no_detection_folder, filename)
            try:
                shutil.move(input_path, new_path)
                print(f"📦 线程{thread_id} 未检测到目标，已移动: {filename} ➜ no_detection")
            except FileNotFoundError:
                print(f"❓ 线程{thread_id} 文件已不存在，可能被其他线程处理: {filename}")


# 主函数
def main():
    # 获取所有文件
    all_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    all_files.sort()
    
    # 将文件列表分成两部分 (或更多，取决于你的CPU核心数)
    num_threads = 2
    threads = []
    chunk_size = len(all_files) // num_threads
    
    for i in range(num_threads):
        start_index = i * chunk_size
        # 最后一个线程处理所有剩余文件
        end_index = (i + 1) * chunk_size if i < num_threads - 1 else len(all_files)
        file_chunk = all_files[start_index:end_index]
        
        thread = threading.Thread(target=worker, args=(file_chunk, i + 1))
        threads.append(thread)
        thread.start()
        
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    print("🎉 所有图片检测并元数据保存完成！")

if __name__ == "__main__":
    main()
