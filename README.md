# NSFW图片处理程序

基于 Flask 的图像处理 Web 应用，支持标签筛选、效果处理和语义搜索。用于对敏感图片进行处理并预览，可在浏览器界面保存结果。具有语义搜索功能，通过中文描述快速找到需要的图片。

## 运行环境
- Windows / PowerShell
- Python 3.10.8

## 使用一键脚本部署
在项目根目录右键文件夹空白处，点击右键菜单中的 `在此处打开 Powershell 窗口`，在打开的新窗口中运行：
```powershell
.\deploy.ps1
```
可使用 `.\deploy.ps1 -SkipModel` 跳过模型下载。

脚本应完成以下工作：
1. 安装 Miniconda 并创建 conda 环境（Python 3.10.8）。
2. 安装 `requirements.txt` 依赖。
3. 下载模型到 `NSFWdetector/`（Falconsai/nsfw_image_detection）。
4. 创建运行所需目录（如 `images/`、`recycle_bin/`、`SafeNet/`、`no_detection/`、`input/` 等）。
5. 可选初始化索引。
6. 启动服务（`app.py`）。

## 手动部署
1. 创建一个Python 3.10.8环境。
2. 运行 `pip install -r requirements.txt`。
3. 下载模型 `https://huggingface.co/Falconsai/nsfw_image_detection/blob/main/model.safetensors` 到 `NSFWdetector/`。

## 一键运行脚本
在项目根目录双击 `run.bat`，或在命令行执行：

```bat
run.bat
```
说明：`run.bat` 使用部署脚本创建的 Miniconda 环境（`.conda_env`）运行。如果尚未部署，请先执行 `.\deploy.ps1`。
然后打开 `localhost:5001`。

## 手动运行流程
默认代码路径下，推荐的处理顺序如下：
1. 在 `./input` 中放置需要处理的图片。
2. 运行 `SafeNet.py`：从 `./input` 读取图片，输出到 `./SafeNet` 或 `./no_detection`，并生成 JSON 标注。`./no_detection` 中将会存放没有检测到目标的图片。
3. 运行 `nsfwDetector.py`：读取 `./SafeNet` 下的图片和 JSON，为每个 JSON 增加 `nsfw` 字段。
4. 将 `./SafeNet` 目录中的图片与 JSON 移动到 `./images`（或修改代码让 `IMAGE_DIR` 指向 `./SafeNet`）。
5. 运行 `index_img.py`：对 `./images` 建立或更新索引。
6. 运行 `app.py`：启动 Web 服务，提供界面与 API。
7. 打开 `localhost:5001`。

说明：
- `index_img.py` 与 `app.py` 默认使用 `images/` 作为图片根目录，因此需要保证最终图片与 JSON 位于 `images/` 下。
- 如果你希望直接使用 `SafeNet/` 作为主目录，请同步修改 `app.py` 与 `index_img.py` 中的 `IMAGE_DIR` / `img_dir`。

## 常见目录
- `input/`：待检测图片
- `SafeNet/`：检测后输出图片与 JSON
- `no_detection/`：未检测到目标的图片
- `images/`：应用主目录（索引与 Web 服务读取）
- `NSFWdetector/`：NSFW 模型目录
- `recycle_bin/`：删除图片的回收站
