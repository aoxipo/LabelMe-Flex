# LabelMe-Flex
[win10、win11] 手动画笔绘制 + 手动多点闭合掩码 + SAM智能分割，三合一交互式轻量化标注工具

 
```
    🎯 鼠标操作：
    🖱️ 左键单击：点击目标进行 SAM 分割
    🖱️ 右键单击：撤销当前类别最后一个分割
    🖱️ 中键按住拖动：手绘掩码（笔刷）
    🖱️ 中键松开：结束绘制并添加掩码
    
    ✏️ 多边形模式：
    🔹 按 P 切换多边形选区模式
    🔹 左键点击添加点，右键闭合区域生成掩码
    
    🎨 缩放操作：
    🔍 滚轮上：放大
    🔎 滚轮下：缩小
    ⭕ 按下 O：还原视图
    
    🏷️ 类别管理：
    ➕ 按 N：切换到下一个类别
    🔙 按 M：返回上一个类别
    
    💡 图像调整：
    🔆 按 E：增强对比度（可多次叠加）
    🔁 按 R：还原原始图像
    按下 '['  or  ']'  调整笔刷半径
    
    💾 其他：
    💾 关闭窗口：自动保存为 LabelMe JSON 和 PNG 掩码
    ❌ 按 ESC：退出程序（关闭所有窗口）
    
    📂 输出路径：
    - JSON 标注文件: {image_dir}/json/
    - 掩码图像文件: {image_dir}/png/
```

# 环境配置
```shell
pip install torch torchvision
pip install opencv-python matplotlib
pip install git+https://github.com/facebookresearch/segment-anything.git
```
# 参数下载
- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

# run

```
python main.py \
  --image_dir ./images \
  --checkpoint ./sam_vit_h_4b8939.pth \
  --model_type vit_h
```

# OPT
继承模式，为了应对3D结构标准，提供上下帧之间的标注点继承，上一帧的标注点会作为下一帧的提示词嵌入到sam结果中
```
inherit_mode=True
```

# 贡献
感谢 SAM 团队开源的算法 https://github.com/facebookresearch/segment-anything.git
