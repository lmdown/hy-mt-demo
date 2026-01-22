# HY-MT1.5 Translator Demo

本项目是对 [Tencent-Hunyuan/HY-MT](https://github.com/Tencent-Hunyuan/HY-MT) 项目的演示应用，基于 HY-MT1.5-1.8B 翻译模型构建，提供简单易用的文本和 PDF 文件翻译功能。

## 功能特性

- 📝 **文本翻译**：支持直接输入文本进行翻译
- 📄 **PDF 翻译**：上传 PDF 文件，自动提取文本并翻译
- 🌍 **多语言支持**：支持 33 种语言的相互翻译
- 💻 **多设备支持**：自动检测并适配 CUDA、MPS、CPU 等设备
- ⚡ **高性能**：基于 1.8B 参数模型，兼顾速度与质量

## 支持的设备

项目会自动检测运行环境并选择最优设备：

- **CUDA**：当检测到 NVIDIA GPU 时，使用 CUDA 加速，采用 float16 精度
- **MPS**：在 macOS 设备上使用 Metal Performance Shaders (MPS) 加速，采用 float16 精度
- **CPU**：在无 GPU 环境下使用 CPU 运行，采用 float32 精度

## 支持的语言

支持以下 33 种语言：

- 中文、英语、日语、韩语
- 法语、德语、西班牙语、葡萄牙语、意大利语
- 俄语、阿拉伯语、土耳其语、泰语、越南语
- 马来语、印尼语、菲律宾语、印地语
- 繁体中文、波兰语、捷克语、荷兰语
- 高棉语、缅甸语、波斯语、古吉拉特语
- 乌尔都语、泰卢固语、马拉地语、希伯来语
- 孟加拉语、泰米尔语、乌克兰语
- 藏语、哈萨克语、蒙古语、维吾尔语、粤语


## 无需命令，一键安装

推荐使用 魔当 (LM Downloader) https://seemts.com/ 一键安装。

## 源码运行

```bash
git clone https://gitee.com/lmdown/hy-mt-demo
cd HY-MT
pip install -r requirements.txt
```

## 模型准备

1. 从 Hugging Face 下载 HY-MT1.5-1.8B 模型：
   ```bash
   huggingface-cli download tencent/HY-MT1.5-1.8B --local-dir models-1.8b
   ```
   也可下载7B模型：
   ```bash
   huggingface-cli download tencent/HY-MT1.5-7B --local-dir models-7b
   ```

2. 可通过环境变量指定模型目录：
   ```bash
   export MODEL_DIR=models-7b
   python app.py
   ```

## 使用方法

启动应用：

```bash
python app.py
```

应用启动后，会在浏览器中打开 Gradio 界面，提供两个翻译选项卡：

1. **文本翻译**：输入文本，选择目标语言，点击翻译按钮
2. **PDF 翻译**：上传 PDF 文件，选择目标语言，点击翻译按钮


## 重要声明
本 Demo 仅为调用示例，底层 HY-MT 模型的使用需遵守《腾讯混元社区许可协议》，且 Demo 的使用范围不得超出 HY-MT 模型的授权边界（如地域限制、使用场景限制）。

1. 本项目仅为腾讯混元HY-MT1.5翻译模型的调用示例（Demo），核心翻译能力依赖[Tencent-Hunyuan/HY-MT](https://github.com/Tencent-Hunyuan/HY-MT)开源项目；
2. 本Demo代码遵循TENCENT HY COMMUNITY LICENSE AGREEMENT，使用本Demo调用HY-MT模型时，需严格遵守《腾讯混元社区许可协议》(https://github.com/Tencent-Hunyuan/HY-MT/blob/main/License.txt)；
3. 禁止欧盟、英国、韩国地区用户使用本Demo调用HY-MT模型，违规使用需自行承担法律责任；
4. 本Demo仅用于学习和非商业测试，不得用于改进非腾讯混元系列AI模型。

## 相关链接

- [HY-MT GitHub 仓库](https://github.com/Tencent-Hunyuan/HY-MT)
- [HY-MT1.5-1.8B 模型](https://huggingface.co/tencent/HY-MT1.5-1.8B)
- [HY-MT1.5-7B 模型](https://huggingface.co/tencent/HY-MT1.5-7B)
