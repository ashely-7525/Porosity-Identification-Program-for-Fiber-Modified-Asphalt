# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:16:17 2025

@author: RDH
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox
from skimage import filters, measure, segmentation
from PIL import ImageTk, Image

# ---------- GUI 主窗口 ----------
class PoreAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("沥青孔隙分析工具")
        self.root.geometry("800x600")

        # 初始化变量
        self.before_image_path = ""
        self.after_image_path = ""

        # ---------- 界面组件 ----------
        # 标题
        Label(root, text="沥青自愈合孔隙分析", font=("Arial", 16)).pack(pady=10)

        # 选择文件区域
        self.frame = Frame(root)
        self.frame.pack(pady=20)

        # 选择拉伸前图像
        Button(self.frame, text="选择拉伸前图像", command=self.load_before_image).grid(row=0, column=0, padx=10)
        self.before_label = Label(self.frame, text="未选择文件", width=50)
        self.before_label.grid(row=0, column=1, padx=10)

        # 选择拉伸后图像
        Button(self.frame, text="选择拉伸后图像", command=self.load_after_image).grid(row=1, column=0, padx=10)
        self.after_label = Label(self.frame, text="未选择文件", width=50)
        self.after_label.grid(row=1, column=1, padx=10)

        # 分析按钮
        Button(root, text="生成热图", command=self.analyze).pack(pady=20)

        # 图像预览区域
        self.preview_frame = Frame(root)
        self.preview_frame.pack()
        self.before_preview = Label(self.preview_frame, text="拉伸前预览")
        self.before_preview.grid(row=0, column=0, padx=10)
        self.after_preview = Label(self.preview_frame, text="拉伸后预览")
        self.after_preview.grid(row=0, column=1, padx=10)

    # ---------- 功能函数 ----------
    def load_before_image(self):
        """加载拉伸前图像"""
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if path:
            self.before_image_path = path
            self.before_label.config(text=path)
            self.show_preview(path, self.before_preview)

    def load_after_image(self):
        """加载拉伸后图像"""
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if path:
            self.after_image_path = path
            self.after_label.config(text=path)
            self.show_preview(path, self.after_preview)

    def show_preview(self, path, label_widget):
        """显示图像预览"""
        img = Image.open(path)
        img.thumbnail((300, 300))  # 缩略图大小
        img_tk = ImageTk.PhotoImage(img)
        label_widget.config(image=img_tk)
        label_widget.image = img_tk  # 保持引用

    def process_image(self, image_path):
        """处理单张图像并生成热图（与之前代码一致）"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"图像加载失败: {image_path}")

        # 预处理
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        thresh = filters.threshold_otsu(blurred)
        binary = blurred > thresh
        cleared = segmentation.clear_border(binary)

        # 孔隙识别
        labels = measure.label(cleared)
        props = measure.regionprops(labels)
        porosity_area = sum(prop.area for prop in props)
        porosity_ratio = porosity_area / (image.shape[0] * image.shape[1])

        # 生成热图
        heatmap = np.zeros_like(image, dtype=np.float32)
        for prop in props:
            heatmap[labels == prop.label] = 1

        return heatmap, porosity_ratio

    def plot_difference(self, heatmap1, heatmap2):
        """绘制差异热图（与之前代码一致）"""
        difference = heatmap1 - heatmap2
        plt.figure(figsize=(10, 8))
        plt.imshow(difference, cmap="coolwarm", interpolation="nearest", vmin=-1, vmax=1)
        plt.colorbar(label="Difference (Before - After)")
        plt.title("孔隙分布差异热图")
        plt.axis("off")
        plt.savefig("pore_difference.png", bbox_inches="tight", dpi=300)
        plt.close()

    def analyze(self):
        """执行分析"""
        if not self.before_image_path or not self.after_image_path:
            messagebox.showerror("错误", "请先选择拉伸前和拉伸后的图像！")
            return

        try:
            # 处理图像
            heatmap_before, ratio_before = self.process_image(self.before_image_path)
            heatmap_after, ratio_after = self.process_image(self.after_image_path)

            # 保存热图
            plt.imsave("heatmap_before.png", heatmap_before, cmap="hot")
            plt.imsave("heatmap_after.png", heatmap_after, cmap="hot")

            # 绘制差异图
            self.plot_difference(heatmap_before, heatmap_after)

            # 显示结果
            result_text = (
                f"拉伸前孔隙率: {ratio_before*100:.2f}%\n"
                f"拉伸后孔隙率: {ratio_after*100:.2f}%\n"
                "热图已保存至当前目录！"
            )
            messagebox.showinfo("分析完成", result_text)

        except Exception as e:
            messagebox.showerror("错误", f"分析失败: {str(e)}")

# ---------- 启动GUI ----------
if __name__ == "__main__":
    root = Tk()
    app = PoreAnalyzerApp(root)
    root.mainloop()