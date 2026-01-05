# delete_image.py
import sys, os, json, faiss
import numpy as np

def delete_image_index(image_path):
    index_file = "image_features.index"
    metadata_file = "index_metadata.json"

    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        print("错误：索引或元数据文件不存在。")
        return

    # 1. 加载索引和元数据
    print(f"正在加载索引...")
    index = faiss.read_index(index_file)
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 2. 查找图片的ID
    path_to_id = {v: k for k, v in metadata["image_map"].items()}
    if image_path not in path_to_id:
        print(f"错误：图片 '{image_path}' 不在索引中。")
        return

    image_id = int(path_to_id[image_path])
    print(f"找到图片 '{image_path}' 对应的ID: {image_id}")

    # 3. 从Faiss索引中删除ID
    index.remove_ids(np.array([image_id], dtype=np.int64))
    print(f"已从Faiss索引中移除ID {image_id}。")

    # 4. 从元数据中删除条目
    del metadata["image_map"][str(image_id)]
    print(f"已从元数据中移除 '{image_path}'。")

    # 5. 保存更改
    faiss.write_index(index, index_file)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print("\n删除成功！索引和元数据已更新。")
    print(f"当前索引总数: {index.ntotal}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python delete_image.py <图片路径>")
        sys.exit(1)
    
    target_path = sys.argv[1]
    delete_image_index(target_path)
