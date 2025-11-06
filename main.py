import os
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.lines import Line2D
from segment_anything import sam_model_registry, SamPredictor
import argparse

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SamInteractiveAnnotator:
    def __init__(self, image_dir, checkpoint, model_type="vit_h", device=None):
        self.image_dir = image_dir
        self.out_dir = os.path.join(image_dir, "json")
        os.makedirs(self.out_dir, exist_ok=True)
        self.out_dir_png = os.path.join(image_dir, "png")
        os.makedirs(self.out_dir_png, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.checkpoint = checkpoint

        print("🚀 加载 SAM 模型中...", self.device)
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        print("✅ 模型加载完成！")

        self.image_list = self._get_image_list()
        self.current_class_id = 0
        self.seg_obj = {}
        self.contrast_level = 0       # 当前对比度增强次数
        self.image_original = None    # 原始图像备份

        self.overlay_image = None     # 实时展示图像
        self.global_mask = None       # 每个像素所属类别ID
         
        

    def print_help(self):
        """打印当前工具的使用说明"""

        msg = """
             ================= 🧭 labelme flex 标注工具使用说明 =================
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

        ============================================================
        """
        
        print(msg)

    def _get_image_list(self):
        img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif"]
        files = [
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if os.path.splitext(f.lower())[1] in img_exts
        ]
        files.sort()
        return files

    @staticmethod
    def id_to_color(class_id):
        np.random.seed(class_id)
        return np.random.randint(0, 255, 3)

    # ====== 图像标注主入口 ======
    def annotate_all(self):
        for idx, path in enumerate(self.image_list):
            print(f"\n[{idx+1}/{len(self.image_list)}] 加载图像: {os.path.basename(path)}")
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

            self.image_original = image.copy()
            self.image = image.copy()

            self.overlay_image = self.image.copy()
           
            self.predictor.set_image(image)
            self._annotate_single_image(image, path)

    # ====== 单图像标注 ======
    def _annotate_single_image(self, image, image_path):
        self.seg_obj.clear()
        self.current_class_id = 0
        self.masks_all = []  # 记录
        self.draw_points = []
        self.is_drawing  = False
        self.temp_mask = None
        self.brush_radius = 10  # 默认笔刷半径

        self.is_polygon_mode = False
        self.poly_points = []
        self.poly_line = None

        self.image = image
        self.fig, self.ax = plt.subplots()
       

        self.ax.imshow(image)
        self.ax.set_title(f"当前类别: 0 | 左键分割, 右键撤销, N/M切换类别, O还原, ESC退出")

        self.fig.canvas.manager.window.setWindowTitle(os.path.basename(image_path))
        self.init_xlim, self.init_ylim = self.ax.get_xlim(), self.ax.get_ylim()

        # 注册事件
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.fig.canvas.mpl_connect("scroll_event", self.onscroll)
        self.fig.canvas.mpl_connect("key_press_event", self.onkey)
        self.cid_move = self.fig.canvas.mpl_connect("motion_notify_event", self.onmove)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self.onrelease)

        plt.show()
        self._save_labelme_json(image_path)
        self._save_png( image_path)

    # ====== 鼠标点击事件 ======
    def onclick(self, event):
        # === 多点闭合选区模式 ===
        if self.is_polygon_mode and event.xdata and event.ydata:
            x, y = int(event.xdata), int(event.ydata)

            # 左键：添加点
            if event.button == 1:
                self.poly_points.append((x, y))
                print(f"🟢 添加点: ({x}, {y})")

                # 更新临时连线
                if self.poly_line:
                    self.poly_line.remove()
                xs, ys = zip(*self.poly_points)
                self.poly_line, = self.ax.plot(xs, ys, "y-", linewidth=1.5)
                self.fig.canvas.draw_idle()

            # 右键：闭合多边形
            elif event.button == 3 and len(self.poly_points) >= 3:
                print("✅ 闭合多边形并生成掩码")

                pts = np.array(self.poly_points, np.int32).reshape((-1, 1, 2))
                mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 1)
                mask = mask > 0
                cid = self.current_class_id
                if cid not in self.seg_obj:
                    self.seg_obj[cid] = {"masks": [], "color": self.id_to_color(cid)}
                self.seg_obj[cid]["masks"].append(mask)
                self.masks_all.append(mask)

                # 应用到显示
                self._add_mask(mask, cid)

                # 清理多边形状态
                self.poly_points.clear()
                if self.poly_line:
                    # self.poly_line.remove()
                    self.poly_line = None
                self.fig.canvas.draw_idle()
            return

        if event.button == 1 and event.xdata and event.ydata:
            x, y = int(event.xdata), int(event.ydata)
            print(f"🟢 点击点: ({x}, {y})")

            masks, scores, _ = self.predictor.predict(
                point_coords=np.array([[x, y]]),
                point_labels=np.array([1]),
                multimask_output=True
            )
            best_mask = masks[np.argmax(scores)]
            self.masks_all.append(best_mask)
             
            cid = self.current_class_id
            if cid not in self.seg_obj:
                self.seg_obj[cid] = {"masks": [], "color": self.id_to_color(cid)}
            self.seg_obj[cid]["masks"].append(best_mask)

            self._add_mask( best_mask, cid)
        

        elif event.button == 3:  # 右键撤销
            cid = self.current_class_id
            if cid in self.seg_obj and self.seg_obj[cid]["masks"]:
                if self.masks_all:
                    self.masks_all.pop()
                print("🟠 撤销上一个分割")
                pop_mask = self.seg_obj[cid]["masks"].pop()
                
                self._undo_last( pop_mask, self.current_class_id)

         # === 中键: 开始绘制区域 ===
        elif event.button == 2:
            print("✏️ 开始手绘掩码区域")
            self.is_drawing = True
            self.brush_radius = getattr(self, "brush_radius", 10)  # 可自定义笔刷大小
            self.temp_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            self.last_pos = (int(event.xdata), int(event.ydata))
            self._draw_circle_on_mask(self.last_pos)
            self._update_overlay_preview()
       
    def onmove(self, event):
        """当鼠标拖动时（仅中键按下）"""
        if self.is_drawing and event.xdata and event.ydata:
            pos = (int(event.xdata), int(event.ydata))
            self._draw_circle_on_mask(pos)
            self.last_pos = pos
            self._update_overlay_preview()


    def onrelease(self, event):
        """当鼠标松开时"""
        if self.is_drawing and event.button == 2:
            self.is_drawing = False
            cid = self.current_class_id

            if cid not in self.seg_obj:
                self.seg_obj[cid] = {"masks": [], "color": self.id_to_color(cid)}
            mask = self.temp_mask > 0
            self.seg_obj[cid]["masks"].append(mask)
            self.masks_all.append(mask)
             
            self._add_mask(mask, cid)
             
            self.temp_mask = None
            
       


    # ====== 滚轮缩放 ======
    def onscroll(self, event):
        base_scale = 1.2
        scale_factor = 1 / base_scale if event.button == "up" else base_scale

        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return

        cur_xlim, cur_ylim = self.ax.get_xlim(), self.ax.get_ylim()
        new_x0 = xdata - (xdata - cur_xlim[0]) * scale_factor
        new_x1 = xdata + (cur_xlim[1] - xdata) * scale_factor
        new_y0 = ydata - (ydata - cur_ylim[0]) * scale_factor
        new_y1 = ydata + (cur_ylim[1] - ydata) * scale_factor

        self.ax.set_xlim(new_x0, new_x1)
        self.ax.set_ylim(new_y0, new_y1)
        self.fig.canvas.draw_idle()

    # ====== 键盘事件 ======
    def onkey(self, event):
        key = event.key.lower()
        if key == "o":
            print("🔵 还原视图")
            self.ax.set_xlim(self.init_xlim)
            self.ax.set_ylim(self.init_ylim)
            self.fig.canvas.draw_idle()
        
        elif key == "r":
            print("🔵 还原图像")
            self.contrast_level = 0
            self.image = self.image_original.copy()
            self.ax.imshow(self.image)
            self.ax.set_xlim(self.init_xlim)
            self.ax.set_ylim(self.init_ylim)
            self.fig.canvas.draw_idle()
            
            self.predictor.set_image(self.image)

        elif key == "n":
            self.current_class_id += 1
            cid = self.current_class_id
            print(f"🟢 切换到类别 {cid}")
            if cid not in self.seg_obj:
                self.seg_obj[cid] = {"masks": [], "color": self.id_to_color(cid)}
            self.ax.set_title(f"当前类别: {cid}")
            self.fig.canvas.draw_idle()

        elif key == "m":
            if self.current_class_id > 0:
                self.current_class_id -= 1
                print(f"🟠 返回类别 {self.current_class_id}")
                self.ax.set_title(f"当前类别: {self.current_class_id}")
                self.fig.canvas.draw_idle()

        elif event.key.lower() == "e":
            # 增强对比度（累积式）
            self.contrast_level += 1
            alpha = 1.2 ** self.contrast_level  # 每次增强20%
            self.image = cv2.convertScaleAbs(self.image_original, alpha=alpha, beta=0)
            print(f"⚡ 对比度增强 ×{self.contrast_level} (alpha={alpha:.2f})")
            self.ax.imshow(self.image)
            self.fig.canvas.draw_idle()
            self.predictor.set_image(self.image)

        elif event.key.lower() == "p":
            self.is_polygon_mode = not self.is_polygon_mode
            mode = "多点选区" if self.is_polygon_mode else "普通模式"
            print(f"🎨 已切换为 {mode}")
            self.poly_points.clear()
            if self.poly_line:
                # self.poly_line.remove()
                self.poly_line = None
            self.fig.canvas.draw_idle()

        elif event.key == "[":
            self.brush_radius = max(1, self.brush_radius - 2)
            print(f"🔹 笔刷半径: {self.brush_radius}")
        elif event.key == "]":
            self.brush_radius += 2
            print(f"🔹 笔刷半径: {self.brush_radius}")
        elif event.key == "escape":
            print("🟥 检测到 ESC，退出所有窗口...")
            plt.close("all")
            exit()


    # ====== 局部更新可视化 ======
    def _add_mask(self, mask, class_id):
        color = self.seg_obj[class_id]["color"]
        if mask.dtype != np.bool:
            raise ValueError("mask must be a boolean array, ", mask.dtype, mask.max())
        
        # 局部更新显示图像
        self.overlay_image[mask] = 0.5 * self.image[mask] + 0.5 * color
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        self.ax.set_xlim( xlim)
        self.ax.set_ylim( ylim)
        self.ax.imshow(self.overlay_image)
        self.ax.set_title(f"当前类别: {self.current_class_id}")
        self.fig.canvas.draw_idle()

    def _undo_last(self, pop_mask, class_id):
        if pop_mask.dtype != np.bool:
            raise ValueError("mask must be a boolean array, ", pop_mask.dtype, pop_mask.max())
        # 局部刷新
        self.overlay_image[pop_mask] = self.image[pop_mask]
        # 🔁 检查是否有其他类别覆盖了这一部分（防止误擦）

        for cid, data in self.seg_obj.items():
            if cid == class_id or not data["masks"]:
                continue
            combined_mask = np.any(np.stack(data["masks"]), axis=0)
            overlap = combined_mask & pop_mask
            if np.any(overlap):
                color = data["color"]
                self.overlay_image[overlap] = 0.5 * self.image[overlap] + 0.5 * color
        
        # 重绘所有已存在类别的叠加色
        # for cid, data in self.seg_obj.items():
        #     mask = self.global_mask == cid
        #     if np.any(mask):
        #         color = data["color"]
        #         self.overlay_image[mask] = 0.5 * self.image[mask] + 0.5 * color

        # 更新显示
        self.ax.clear()
        self.ax.imshow(self.overlay_image)
        self.ax.set_title(f"当前类别: {self.current_class_id}")
        self.fig.canvas.draw_idle()

    # ====== 画笔绘制 ======
    def _draw_circle_on_mask(self, center):
        """在临时掩码上绘制一个圆"""
        cv2.circle(self.temp_mask, center, self.brush_radius, 1, -1)

    def _update_overlay_preview(self):
        """实时预览绘制结果（半透明叠加）"""
    
        preview = self.overlay_image.copy()
        color = self.id_to_color(self.current_class_id)
        preview[self.temp_mask > 0] = 0.5 * preview[self.temp_mask > 0] + 0.5 * color
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        self.ax.set_xlim( xlim)
        self.ax.set_ylim( ylim)
        self.ax.imshow(preview)
        self.ax.set_title(f"当前类别: {self.current_class_id}（笔刷半径={self.brush_radius}）")
        self.fig.canvas.draw_idle()

    # ====== 保存为 LabelMe JSON ======
    def _save_labelme_json(self, image_path):
        if not self.seg_obj:
            print("⚠️ 未标注任何物体，跳过保存。")
            return

        shapes = []
        h, w = self.image.shape[:2]
        for cid, data in self.seg_obj.items():
            if not data["masks"]:
                continue

            combined_mask = np.any(np.stack(data["masks"]), axis=0)
            contours, _ = cv2.findContours(
                combined_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                # === 🔧 优化：简化轮廓点 ===
                epsilon = 0.001 * cv2.arcLength(contour, True)  # 0.5% 的周长作为误差容忍
                approx = cv2.approxPolyDP(contour, epsilon, True)

                pts = approx.squeeze(1).tolist()
                if len(pts) < 3:  # 排除无效多边形
                    continue

                shapes.append({
                    "label": str(cid),
                    "points": pts,
                    "shape_type": "polygon",
                    "flags": {}
                })

        labelme_json = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(image_path),
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w
        }

        save_path = os.path.join(
            self.out_dir,
            os.path.splitext(os.path.basename(image_path))[0] + ".json"
        )
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(labelme_json, f, indent=2, ensure_ascii=False)

        print(f"✅ 已保存（简化多边形，减少点数）: {save_path}")

    def _save_png(self, image_path):
        if not self.seg_obj:
            print("⚠️ 未标注任何物体，跳过保存PNG。")
            return

        h, w = self.image_original.shape[:2]
        mask_img = np.zeros((h, w), dtype=np.uint8)
        save_path = os.path.join(
            self.out_dir_png,
            os.path.splitext(os.path.basename(image_path))[0] + ".png"
        )

        for cid, data in self.seg_obj.items():
            if not data["masks"]:
                continue
            combined_mask = np.any(np.stack(data["masks"]), axis=0)
            mask_img[combined_mask] = cid + 1

        cv2.imwrite(save_path, mask_img)
        print(f"✅ 已保存(png 掩码): {save_path}")



# ==== 运行 ====
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="导出SAM2为onnx文件")
    parser.add_argument("--imgdir",type=str,default=r"D:\work\Code\auto_seg\data",required=False,help="path")
    parser.add_argument("--modeltype",type=str,default=r"sam_vit_h_4b8939.pth",required=False,help="vit_h")
    parser.add_argument("--checkpoint",type=str,default="vit_h",required=False,help="*.pt")
    args = parser.parse_args()

    annotator = SamInteractiveAnnotator(
        # image_dir=r"D:\dataset\Focus",
        image_dir= args.imgdir,
        # image_dir=r"D:\dataset\Focus\best\merge",
        checkpoint= args.checkpoint,
        model_type= args.modeltype,
    )
    annotator.print_help()
    annotator.annotate_all()

