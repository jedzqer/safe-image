import os
from pathlib import Path

def check_image_json_pairs(directory="."):
    """
    检测指定目录下images文件夹内的图片文件和json文件的配对情况
    包括缺失的json和多余的json
    
    Args:
        directory: 要检查的根目录，默认为当前目录
    """
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    # 构建images文件夹路径
    images_path = Path(directory) / "images"
    
    if not images_path.exists():
        print(f"❌ 未找到images文件夹: {images_path}")
        return
    
    if not images_path.is_dir():
        print(f"❌ images不是一个文件夹: {images_path}")
        return
    
    print(f"🔍 正在检查目录: {images_path}")
    print("=" * 50)
    
    # 收集所有图片文件和JSON文件
    image_files = set()
    json_files = set()
    
    for file_path in images_path.iterdir():
        if file_path.is_file():
            if file_path.suffix.lower() in image_extensions:
                image_files.add(file_path.stem)  # 文件名（不含扩展名）
            elif file_path.suffix.lower() == '.json':
                json_files.add(file_path.stem)
    
    # 分析配对情况
    perfect_pairs = []      # 完美配对
    missing_json = []       # 缺少JSON
    orphan_json = []        # 多余的JSON
    
    # 检查每个图片文件
    for img_name in image_files:
        if img_name in json_files:
            perfect_pairs.append(img_name)
            print(f"✅ {img_name}.* -> {img_name}.json")
        else:
            missing_json.append(img_name)
            print(f"❌ {img_name}.* -> 缺少 {img_name}.json")
    
    # 检查多余的JSON文件
    for json_name in json_files:
        if json_name not in image_files:
            orphan_json.append(json_name)
            print(f"🗑️  {json_name}.json -> 没有对应的图片文件")
    
    # 输出统计结果
    print("\n" + "=" * 50)
    print(f"📊 检查完成！")
    print(f"🖼️  总图片数量: {len(image_files)}")
    print(f"📄 总JSON数量: {len(json_files)}")
    print(f"✅ 完美配对: {len(perfect_pairs)} 个")
    print(f"❌ 缺少JSON: {len(missing_json)} 个")
    print(f"🗑️  多余JSON: {len(orphan_json)} 个")
    
    # 详细列表
    if missing_json:
        print(f"\n🚨 缺少JSON文件的图片:")
        for img in sorted(missing_json):
            print(f"   - {img}.*")
    
    if orphan_json:
        print(f"\n🗑️  多余的JSON文件:")
        for json_name in sorted(orphan_json):
            print(f"   - {json_name}.json")
    
    if not missing_json and not orphan_json:
        print(f"\n🎉 完美！所有文件都正确配对！")
    
    # 返回详细结果
    return {
        'total_images': len(image_files),
        'total_json': len(json_files),
        'perfect_pairs': perfect_pairs,
        'missing_json': missing_json,
        'orphan_json': orphan_json
    }

def show_file_details(directory="."):
    """
    显示文件夹中所有图片和JSON文件的详细信息
    """
    images_path = Path(directory) / "images"
    if not images_path.exists():
        return
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    print(f"\n📁 文件夹详情: {images_path}")
    print("-" * 30)
    
    all_files = list(images_path.iterdir())
    all_files.sort()
    
    for file_path in all_files:
        if file_path.is_file():
            if file_path.suffix.lower() in image_extensions:
                print(f"🖼️  {file_path.name}")
            elif file_path.suffix.lower() == '.json':
                print(f"📄 {file_path.name}")

if __name__ == "__main__":
    print("🎨 图片-JSON文件配对检测器 (增强版)")
    print("=" * 50)
    
    # 检查当前目录
    result = check_image_json_pairs(".")
    
    # 询问是否显示详细文件列表
    print("\n" + "=" * 30)
    show_details = input("🤔 是否显示文件夹中的所有文件？(y/N): ").strip().lower()
    
    if show_details in ['y', 'yes', '是', 'Y']:
        show_file_details(".")
    
    print(f"\n👋 检查完成！")
