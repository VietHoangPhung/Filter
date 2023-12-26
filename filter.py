import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

print(os.getcwd())
class ImageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Filter App")
        self.image_path = None
        self.original_image = None
        self.filtered_image = None
        self.kernel_size_var = tk.IntVar()
        self.threshold1_var = tk.StringVar()
        self.threshold2_var = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        # Button to open an image
        self.open_button = tk.Button(self.root, text="Open Image", command=self.open_image)
        self.open_button.pack(pady=10)

        # Button to display original image
        self.display_original_button = tk.Button(self.root, text="Display Original Image", command=self.display_original)
        self.display_original_button.pack(pady=5)
        self.display_original_button['state'] = 'disabled'

        # Entry for Canny edge detection thresholds
        self.threshold1_label = tk.Label(self.root, text="Lower threshold:")
        self.threshold1_label.pack()
        self.threshold1_entry = tk.Entry(self.root, textvariable=self.threshold1_var)
        self.threshold1_entry.pack()

        self.threshold2_label = tk.Label(self.root, text="Upper threshold:")
        self.threshold2_label.pack()
        self.threshold2_entry = tk.Entry(self.root, textvariable=self.threshold2_var)
        self.threshold2_entry.pack()

        # Dropdown menu to select filter type
        filter_options = ["Trung Bình (Average)", "Trung Vị (Median)", "Max", "Min", "Histogram Equalization", "Otsu"]
        self.filter_var = tk.StringVar()
        self.filter_var.set(filter_options[0])
        self.filter_dropdown = tk.OptionMenu(self.root, self.filter_var, *filter_options)
        self.filter_dropdown.pack(pady=5)

        # Entry for kernel size
        self.kernel_size_label = tk.Label(self.root, text="Kernel Size:")
        self.kernel_size_label.pack()
        self.kernel_size_entry = tk.Entry(self.root, textvariable=self.kernel_size_var)
        self.kernel_size_entry.pack()

        # Apply Filter button
        self.apply_button = tk.Button(self.root, text="Apply Filter", command=self.apply_filter)
        self.apply_button.pack(pady=10)
        self.apply_button['state'] = 'disabled'

        # Button to apply Canny edge detection
        self.canny_button = tk.Button(self.root, text="Apply Canny Edge Detection", command=self.apply_canny_edge_detection)
        self.canny_button.pack(pady=10)
        self.canny_button['state'] = 'disabled'

        # Button to save filtered image
        self.save_button = tk.Button(self.root, text="Save Filtered Image", command=self.save_filtered_image)
        self.save_button.pack(pady=5)
        self.save_button['state'] = 'disabled'

        # Display image panel
        self.image_panel = tk.Label(self.root)
        self.image_panel.pack(padx=10, pady=10)

    def open_image(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
            if file_path:
                image_size_kb = os.path.getsize(file_path) / 1024  # Size in KB
                max_image_size_kb = 500

                if image_size_kb > max_image_size_kb:
                    messagebox.showerror("Error", f"Selected image size exceeds the limit ({max_image_size_kb} KB). Please choose a smaller image.")
                    return
                self.image_path = file_path
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                self.original_image = Image.fromarray(image)
                self.display_image(self.original_image)
                self.display_original_button['state'] = 'normal'
                self.apply_button['state'] = 'normal'
                self.canny_button['state'] = 'normal'
        except:
            messagebox.showerror("Error:", "Please choose image in the same folder with the running file or move target image to this folder")
            self.image_path = None
            self.original_image = None
            self.filtered_image = None
            self.save_button['state'] = 'disabled'
            self.canny_button['state'] = 'disabled'
            self.apply_button['state'] = 'disabled'

    def display_original(self):
        if self.original_image:
            self.display_image(self.original_image)

    def apply_filter(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        filter_type = self.filter_var.get()
        
        use_kernel_size = ["Trung Bình (Average)", "Trung Vị (Median)", "Max", "Min"]
        if filter_type in use_kernel_size:
            try:
                kernel_size = self.kernel_size_var.get()
                if kernel_size > 100:
                    messagebox.showwarning("Error", "Too big kernel size, this device cannot handle.\nYou are not wrong but this is limited to make sure that the program works well..." )
                elif kernel_size == 0:
                    messagebox.showerror("Error", "Please enter kernel size as a positive integer")  
                else:
                    if filter_type == "Trung Bình (Average)":
                        self.filtered_image = Image.fromarray(self.average_filter(image, kernel_size))
                    elif filter_type == "Trung Vị (Median)":
                        self.filtered_image = Image.fromarray(self.median_filter(image, kernel_size))
                    elif filter_type == "Max":
                        self.filtered_image = Image.fromarray(self.max_filter(image, kernel_size))
                    elif filter_type == "Min":
                        self.filtered_image = Image.fromarray(self.min_filter(image, kernel_size))
            except:
                messagebox.showerror("Error", "Please enter kernel size as a positive integer")
        elif filter_type == "Histogram Equalization":
            self.filtered_image = Image.fromarray(self.histogram_equalization(image))
        elif filter_type == "Otsu":
            self.filtered_image = Image.fromarray(self.apply_otsu(image))
        try:
            self.display_image(self.filtered_image)
            self.save_button['state'] = 'normal'
        except:
            pass
            

    def apply_canny_edge_detection(self):
        try:
            lower = int(self.threshold1_var.get())
            upper = int(self.threshold2_var.get())
            if lower < 0 or upper < 0 or lower >= upper:
                messagebox.showerror("Error", "Enter positive values.\n Lower value must be smaller than upper value")
            else:
                edges = cv2.Canny(np.array(self.original_image), lower, upper)
                self.filtered_image = Image.fromarray(edges)
                self.display_image(self.filtered_image)
                self.save_button['state'] = 'normal'
        except:
            messagebox.showerror("Error", "Enter threshold to use this function")

    def save_filtered_image(self):
        if self.filtered_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                self.filtered_image.save(file_path)
                messagebox.showinfo("Success", "Filtered image saved successfully.")

    def average_filter(self, image, kernel_size):
        filtered_image = self.Trung_Binh(image, kernel_size)
        return filtered_image

    def median_filter(self, image, kernel_size):
        filtered_image = self.Trung_Vi(image, kernel_size)
        return filtered_image

    def max_filter(self, image, kernel_size):
        filtered_image = self.loc_max(image, kernel_size)
        return filtered_image

    def min_filter(self, image, kernel_size):
        filtered_image = self.loc_min(image, kernel_size)
        return filtered_image

    def histogram_equalization(self, img):
        hist = cv2.calcHist([img],[0],None,[256],ranges=[0,256])
    
        m, n = img.shape[:2]
        pdf = hist / (m*n)

        cdf = np.cumsum(pdf)

        s = np.round(255*cdf).astype("uint8")

        equ = np.zeros_like(img)
        equ = s[img]

        return equ

    def apply_otsu(self, img):
        total_pixels = img.size
        hist, bins = np.histogram(img, bins=np.arange(257))

        opt_thres = -1
        var = -1

        for k in bins[1:-1]:
            p1 = np.sum(hist[:k]) / total_pixels    
            m1 = np.sum(np.arange(k+1) * hist[:k+1]) / np.sum(hist[:k+1])
            m2 = np.sum(np.arange(k+1, 256) * hist[k+1:]) / np.sum(hist[k+1:])
            var_temp = p1 * (1-p1) * (m1-m2)**2

            if var < var_temp:
                var = var_temp
                opt_thres = k

        thresholed_img = img.copy()
        thresholed_img[img > opt_thres] = 255
        thresholed_img[img <= opt_thres] = 0
        return thresholed_img

    def display_image(self, img):
        try:
            img.thumbnail((400, 400))
            img = ImageTk.PhotoImage(img)
            self.image_panel.configure(image=img)
            self.image_panel.image = img
        except:
            self.image_panel.configure(image=None)

    def Trung_Binh(self, img_matrix, kernel_size):
        height, width = img_matrix.shape
        f_height, f_width = kernel_size, kernel_size
        filtered_image = np.copy(img_matrix)

        # Tính toán padding cho ảnh
        pad = kernel_size // 2

        # Duyệt qua từng pixel trong ảnh
        padded_img = np.pad(img_matrix, pad, mode='constant')

        for i in range(height):
            for j in range(width):
                # lấy ma trận con xung quanh pixel hiện tại
                neighborhood = padded_img[i:i + f_height, j:j + f_width]
                # Tính trung bình của ma trận con
                average = np.mean(neighborhood, axis=(0, 1))
                # Gán giá trị trung bình cho pixel hiện tại trong ảnh lọc
                filtered_image[i, j] = average

        return filtered_image

    def Trung_Vi(self, img, kernel_size):
        h, w = img.shape
        restored_img = np.zeros_like(img)
        for i in range(h - kernel_size // 2):
            for j in range(w - kernel_size // 2):
                neighbors = [img[i, j]]
                for s in range(1, (kernel_size // 2) + 1):
                    neighbors.append(img[i - s, j - s])
                    neighbors.append(img[i - s, j])
                    neighbors.append(img[i - s, j + s])
                    neighbors.append(img[i, j - 1])
                    neighbors.append(img[i, j + 1])
                    neighbors.append(img[i + s, j - s])
                    neighbors.append(img[i + s, j])
                    neighbors.append(img[i + s, j + s])
                sorted_neighbors = np.sort(np.array(neighbors))
                restored_img[i, j] = np.median(sorted_neighbors)
        return restored_img

    def loc_max(self, img, kernel_size):
        h, w = img.shape
        restored_img = np.zeros_like(img)
        for i in range(h - kernel_size // 2):
            for j in range(w - kernel_size // 2):
                neighbors = [img[i, j]]
                for s in range(1, (kernel_size // 2) + 1):
                    neighbors.append(img[i - s, j - s])
                    neighbors.append(img[i - s, j])
                    neighbors.append(img[i - s, j + s])
                    neighbors.append(img[i, j - 1])
                    neighbors.append(img[i, j + 1])
                    neighbors.append(img[i + s, j - s])
                    neighbors.append(img[i + s, j])
                    neighbors.append(img[i + s, j + s])
                restored_img[i, j] = max(neighbors)
        return restored_img

    def loc_min(self, img, kernel_size):
        h, w = img.shape
        restored_img = np.zeros_like(img)
        for i in range(h - kernel_size // 2):
            for j in range(w - kernel_size // 2):
                neighbors = [img[i, j]]
                for s in range(1, (kernel_size // 2) + 1):
                    neighbors.append(img[i - s, j - s])
                    neighbors.append(img[i - s, j])
                    neighbors.append(img[i - s, j + s])
                    neighbors.append(img[i, j - 1])
                    neighbors.append(img[i, j + 1])
                    neighbors.append(img[i + s, j - s])
                    neighbors.append(img[i + s, j])
                    neighbors.append(img[i + s, j + s])
                restored_img[i, j] = min(neighbors)
        return restored_img

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageFilterApp(root)
    root.mainloop()
