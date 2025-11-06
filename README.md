# LabelMe-Flex
[win10、win11] 手动画笔绘制 + 手动多点闭合掩码 + SAM智能分割，三合一交互式轻量化标注工具

结果会在图像路径下创建json文件夹和png文件夹
```
<image_dir>/
├── json/    # LabelMe 格式 JSON 文件
└── png/     # 掩码图像（按类别 ID 编码）
```
| 操作        | 功能           |
| --------- | ------------ |
| 左键单击      | SAM 分割点      |
| 右键单击      | 撤销当前类别最后一个掩码 |
| 中键按住拖动    | 手绘掩码（笔刷模式）   |
| 中键松开      | 结束笔刷绘制       |
| 滚轮        | 缩放视图         |
| `O`       | 还原视图         |
| `N` / `M` | 切换类别 ID      |
| `E` / `R` | 增强或还原对比度     |
| `[` / `]` | 调整笔刷半径       |
| `P`       | 切换多边形模式      |
| `Q`       | 结束当前图片，并保存标注结果    |
| `ESC`     | 退出程序         |


# 环境配置
```shell
pip install torch torchvision
pip install opencv-python matplotlib
pip install git+https://github.com/facebookresearch/segment-anything.git
```

# run

```
python main.py \
  --image_dir ./images \
  --checkpoint ./sam_vit_h_4b8939.pth \
  --model_type vit_h
```
