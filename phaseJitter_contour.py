import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog

class ContourJitteringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Phase Jittering for Contours")
        self.root.geometry("1000x600")
        
        # 상단 프레임 (컨트롤 패널)
        self.top_frame = ttk.Frame(self.root)
        self.top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 이미지 로드 버튼
        self.load_button = ttk.Button(self.top_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        # 지터링 버튼
        self.jitter_button = ttk.Button(self.top_frame, text="Jittering", command=self.apply_jittering)
        self.jitter_button.pack(side=tk.LEFT, padx=5)
        self.jitter_button.config(state='disabled')
        
        # 지터링 강도 콤보박스
        ttk.Label(self.top_frame, text="Jittering Ratio:").pack(side=tk.LEFT, padx=(20, 5))
        self.jitter_amount = tk.StringVar()
        self.jitter_values = ["0.00001", "0.00005", "0.0001", "0.0005", "0.001", "0.005", "0.01", "0.05", "0.1"]
        self.jitter_combo = ttk.Combobox(self.top_frame, textvariable=self.jitter_amount, values=self.jitter_values, state="readonly", width=5)
        self.jitter_combo.current(2)  # 기본값 0.3
        self.jitter_combo.pack(side=tk.LEFT, padx=5)
        
        # 이미지 표시 영역
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Matplotlib 설정
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 초기화
        self.contour = None
        self.complex_contour = None
        self.img = None
        self.status_label = ttk.Label(self.root, text="Status: Load Your Image.")
        self.status_label.pack(anchor=tk.W, padx=10, pady=5)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("이미지 파일", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
        )
        
        if not file_path:
            return
        
        # 이미지 로드 및 전처리
        self.img = cv2.imread(file_path)
        if self.img is None:
            self.status_label.config(text="status: Fail to loading image")
            return
            
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            self.status_label.config(text="status: Fail to finding Contours")
            return
            
        # 가장 큰 윤곽선 찾기
        self.contour = max(contours, key=cv2.contourArea)
        
        # 복소수 형태로 윤곽선 변환 (x + iy)
        self.complex_contour = self.contour[:, 0, 0] + 1j * self.contour[:, 0, 1]
        
        # 원본 윤곽선 표시
        self.update_plot(original_only=True)
        self.jitter_button.config(state='normal')
        self.status_label.config(text=f"status: Loading image OK. Contius Points: {len(self.contour)}")
    
    def apply_jittering(self):
        if self.complex_contour is None:
            return
        
        # 지터링 강도 가져오기
        jitter_amount = float(self.jitter_amount.get())
        
        # 윤곽선의 푸리에 변환
        fourier_desc = np.fft.fft(self.complex_contour)
        
        # 위상에 랜덤 지터링 적용
        phase_random = np.exp(1j * 2 * np.pi * np.random.rand(*fourier_desc.shape) * jitter_amount)
        fourier_jittered = fourier_desc * phase_random
        
        # 역변환으로 지터링된 윤곽선 복원
        contour_jittered = np.fft.ifft(fourier_jittered)
        
        # 복소수에서 x, y 좌표 추출
        x_jittered = contour_jittered.real
        y_jittered = contour_jittered.imag
        
        # 플롯 업데이트
        self.update_plot(jittered_x=x_jittered, jittered_y=y_jittered)
        self.status_label.config(text=f"status: Jittering OK (Ratio: {jitter_amount})")
    
    def update_plot(self, jittered_x=None, jittered_y=None, original_only=False):
        # 플롯 초기화
        self.ax1.clear()
        self.ax2.clear()
        
        # 원본 윤곽선 표시
        x_orig = self.contour[:, 0, 0]
        y_orig = self.contour[:, 0, 1]
        self.ax1.plot(x_orig, y_orig, 'b-', linewidth=2)
        self.ax1.set_title('Original Contour')
        self.ax1.axis('equal')
        
        # 지터링된 윤곽선 표시 (있는 경우에만)
        if not original_only and jittered_x is not None and jittered_y is not None:
            self.ax2.plot(jittered_x, jittered_y, 'r-', linewidth=2)
            self.ax2.set_title('Jittered Contour')
        else:
            self.ax2.set_title('Press Jittering Button')
        
        self.ax2.axis('equal')
        
        # 두 축의 범위를 맞춤
        x_min, x_max = min(x_orig), max(x_orig)
        y_min, y_max = min(y_orig), max(y_orig)
        
        # 여백 추가
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        
        self.ax1.set_xlim(x_min - x_margin, x_max + x_margin)
        self.ax1.set_ylim(y_min - y_margin, y_max + y_margin)
        self.ax2.set_xlim(x_min - x_margin, x_max + x_margin)
        self.ax2.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # 캔버스 업데이트
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ContourJitteringGUI(root)
    root.mainloop()
