import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, RadioButtons, CheckButtons
from tkinter import filedialog
import tkinter as tk
from scipy.signal import savgol_filter
import h5py 
import os
from PIL import Image
import tifffile as tiff 


class SpectrumViewer:
    def __init__(self):
        self.data = None
        self.points = []
        self.colors = ['black', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
        self.bin_size = 1
        self.spectrum_cursor = None  # 스펙트럼 커서 라인
        self.cursor_band = 0  #
        self.markers = []
        self.rectangles = []
        self.rgb_bands = [1016,1073,1100]
        self.initialdir=r'C:\Users\User\Documents\Python Scripts\denoised'
        self.apply_snv = False  # SNV 적용 여부
        self.autoscale = True  # Autoscale 기본값 ON
        self.loaded_filename = True # 로드된 파일명 저장
        self.export_filename = 'Name' # 초기 파일 내임
        self.rgb_cursors = [None, None, None]  # R, G, B 커서 라인
        self.dragging_cursor = None  # 현재 드래그 중인 커서
        
        # Figure 생성
        self.fig = plt.figure(figsize=(16, 9))
        gs = self.fig.add_gridspec(3, 3, height_ratios=[3, 2, 0.3], width_ratios=[2, 2, 1])
        
        # RGB 이미지 (좌측 상단)
        self.ax_img = self.fig.add_subplot(gs[0, 0])
        self.ax_img.set_title('No image loaded - Click "Load Image"')
        self.ax_img.axis('off')
        self.im = None
        
        # 파일명 표시용 텍스트 (이미지 하단)
        self.filename_text = self.ax_img.text(2.5, -0.05, '', 
                                             transform=self.ax_img.transAxes,
                                             ha='center', va='top',
                                             fontsize=10, style='italic',
                                             bbox=dict(boxstyle='round,pad=0.5', 
                                                      facecolor='wheat', alpha=0.8))
        
        # 스펙트럼 그래프 (중앙 상단)
        self.ax_spec = self.fig.add_subplot(gs[0, 1])
        self.spec_lines = []
        self.ax_spec.set_xlabel('Band Number')
        self.ax_spec.set_ylabel('Intensity')
        self.ax_spec.set_title('Multi-point Spectra')
        self.ax_spec.grid(True)
        
        # 밴드 이미지 (우측 상단)
        self.ax_band = self.fig.add_subplot(gs[0, 2])
        self.ax_band.set_title('No image loaded')
        self.ax_band.axis('off')
        self.band_img = None
        
        # 포인트 리스트 텍스트 (좌측 중간)
        self.ax_info = self.fig.add_subplot(gs[1, 0])
        self.ax_info.axis('off')
        self.point_text = self.ax_info.text(0.05, 0.95, '', fontsize=10, verticalalignment='top',
                                           family='monospace')
        self.update_point_list()
        
        # 빈 공간 (중앙 중간)
        self.ax_empty = self.fig.add_subplot(gs[1, 1])
        self.ax_empty.axis('off')
        
        # 컨트롤 패널 (우측 중간)
        self.ax_control = self.fig.add_subplot(gs[1, 2])   
        self.ax_control.axis('off')
        
        # Load Image 버튼
        ax_load = plt.axes([0.8, 0.35, 0.15, 0.05])
        self.btn_load = Button(ax_load, 'Load Image')
        self.btn_load.on_clicked(self.load_image)

        # RGB 밴드 입력 (R, G, B)
        # RGB 라벨 먼저 추가 (figure 좌표계 사용)
        self.fig.text(0.42, 0.25, 'False Color (R,G,B):', fontsize=14)
        
        ax_r = plt.axes([0.42, 0.2, 0.04, 0.03])
        ax_g = plt.axes([0.47, 0.2, 0.04, 0.03])
        ax_b = plt.axes([0.52, 0.2, 0.04, 0.03])
        
        self.tb_r = TextBox(ax_r, '', initial=str(self.rgb_bands[0]))
        self.tb_g = TextBox(ax_g, '', initial=str(self.rgb_bands[1]))
        self.tb_b = TextBox(ax_b, '', initial=str(self.rgb_bands[2]))
        
        self.tb_r.on_submit(lambda text: self.update_rgb_bands(0, text))
        self.tb_g.on_submit(lambda text: self.update_rgb_bands(1, text))
        self.tb_b.on_submit(lambda text: self.update_rgb_bands(2, text))
        
        # SNV 체크박스 추가
        ax_snv = plt.axes([0.6, 0.35, 0.15, 0.05])
        self.check_snv = CheckButtons(ax_snv, ['SNV Normalization'], [False])
        self.check_snv.on_clicked(self.toggle_snv)
        
        # Autoscale 체크박스 추가
        ax_autoscale = plt.axes([0.6, 0.3, 0.15, 0.05])
        self.check_autoscale = CheckButtons(ax_autoscale, ['Autoscale Y-axis'], [True])
        self.check_autoscale.on_clicked(self.toggle_autoscale)
        
        # 비닝 크기 입력
        ax_bin = plt.axes([0.8, 0.25, 0.15, 0.04])
        self.bin_textbox = TextBox(ax_bin, 'Bin Size:', initial='1')
        self.bin_textbox.on_submit(self.update_bin_size)
        
        # Clear 버튼
        ax_clear = plt.axes([0.6, 0.25, 0.15, 0.05])
        self.btn_clear = Button(ax_clear, 'Clear All')
        self.btn_clear.on_clicked(self.clear_all)
        
        # 파일명 입력 TextBox
        ax_filename = plt.axes([0.8, 0.15, 0.15, 0.04])
        self.tb_filename = TextBox(ax_filename, 'Filename:', initial=self.export_filename)
        self.tb_filename.on_submit(self.update_filename)
        
        # Export 버튼
        ax_export = plt.axes([0.6, 0.15, 0.15, 0.05])
        self.btn_export = Button(ax_export, 'Export Spectra')
        self.btn_export.on_clicked(self.export_spectra)
        
        # 슬라이더 (하단)
        ax_slider = plt.axes([0.1, 0.08, 0.8, 0.03])
        self.slider = Slider(ax_slider, 'Band', 0, 1, valinit=0, valstep=1)
        self.slider.on_changed(self.update_band)
        self.slider.set_active(False)
        
        # 드래그 관련 변수
        self.dragging = False
        self.drag_point_idx = None
        
        # 이벤트 연결
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release) # 
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion) # 마우스 움직일때
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        ax_save_falsecolor = plt.axes([0.42, 0.15, 0.14, 0.04])
        self.btn_save_falsecolor = Button(ax_save_falsecolor, 'Save False Color')
        self.btn_save_falsecolor.on_clicked(self.save_falsecolor_image)

                
    def update_rgb_cursors(self):  # ✅ 파라미터 제거
        """RGB 커서 라인 업데이트"""
        if self.data is None:
            return
            
        colors = ['red', 'green', 'blue']
        labels = ['R', 'G', 'B']
        
        # 기존 커서 제거
        for cursor in self.rgb_cursors:
            if cursor:
                cursor.remove()
        
        # 새 커서 추가
        for i in range(3):
            band = self.rgb_bands[i]
            self.rgb_cursors[i] = self.ax_spec.axvline(
                band, color=colors[i], linestyle='-', linewidth=2, 
                alpha=0.8, label=f'{labels[i]}: {band}'
            )

    def on_motion(self, event):
        """마우스 이동 - 드래그 중 포인트/커서 이동"""
        # 포인트 드래그
        if self.dragging and self.drag_point_idx is not None:
            if event.inaxes == self.ax_img and event.xdata and event.ydata:
                x, y = int(event.xdata), int(event.ydata)
                
                if 0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]:
                    # 포인트 위치 업데이트
                    old_x, old_y, label = self.points[self.drag_point_idx]
                    self.points[self.drag_point_idx] = (x, y, label)
                    
                    # 실시간 업데이트
                    self.plot_all_spectra()
                    self.update_point_list()
        
        # RGB 커서 드래그
        elif self.dragging_cursor is not None:
            if event.inaxes == self.ax_spec and event.xdata is not None:
                band = int(round(event.xdata))
                if 0 <= band < self.data.shape[2]:
                    # RGB 밴드 업데이트
                    self.rgb_bands[self.dragging_cursor] = band
                    
                    # TextBox 업데이트
                    if self.dragging_cursor == 0:
                        self.tb_r.set_val(str(band))
                    elif self.dragging_cursor == 1:
                        self.tb_g.set_val(str(band))
                    else:
                        self.tb_b.set_val(str(band))
                    
                    if not hasattr(event, 'button') or event.button is None:
                        if self.dragging_cursor is not None:
                            print("Ending cursor drag via motion")
                            self.dragging_cursor = None
                        if self.dragging:
                            self.dragging = False
                            self.drag_point_idx = None
                        return

                    # 커서와 이미지 업데이트
                    self.update_rgb_cursors()
                    self.update_rgb_image()



    def on_release(self, event):
        """마우스 버튼 릴리즈 - 드래그 종료"""
        if self.dragging:
            self.dragging = False
            self.drag_point_idx = None
        
        # ✅ RGB 커서 드래그 종료 추가
        if self.dragging_cursor is not None:
            self.dragging_cursor = None
            print("RGB cursor drag ended")



    def update_filename_display(self):
        """이미지 하단에 파일명 표시"""
        if self.loaded_filename:
            self.filename_text.set_text(f'File: {self.loaded_filename}')
            self.filename_text.set_visible(True)
        else:
            self.filename_text.set_visible(False)
        self.fig.canvas.draw_idle()
    
    def apply_snv_normalization(self, spectrum):
        """SNV 정규화 적용"""
        mean = np.mean(spectrum)
        std = np.std(spectrum)
        if std == 0:
            return spectrum - mean
        return (spectrum - mean) / std
    
    def toggle_snv(self, label):
        """SNV On/Off 토글"""
        self.apply_snv = not self.apply_snv
        print(f"SNV Normalization: {'ON' if self.apply_snv else 'OFF'}")
        if self.points:
            self.plot_all_spectra()
    
    def toggle_autoscale(self, label):
        """Autoscale On/Off 토글"""
        self.autoscale = not self.autoscale
        print(f"Autoscale Y-axis: {'ON' if self.autoscale else 'OFF'}")
        if self.points:
            self.plot_all_spectra()
    
    def load_image(self, event):
        try:
            # 별도의 root 생성 없이 바로 파일 다이얼로그 호출
            data_files = filedialog.askopenfilename(
                title='Select Hyperspectral Image',
                initialdir=self.initialdir,
                filetypes=[
                    ('Hyperspectral files', '*.mat *.h5 *.hdf5 *.tiff *.tif'),
                    ('MAT files', '*.mat'),
                    ('HDF5 files', '*.h5 *.hdf5'),
                    ('TIFF files', '*.tiff *.tif'),
                    ('All files', '*.*')
                ]
            )
            
            if not data_files:
                print("No file selected")
                return
            
            # 파일명 정보 업데이트
            self.loaded_filename = os.path.basename(data_files)
            base_name = os.path.splitext(self.loaded_filename)[0]
            self.export_filename = base_name
            self.tb_filename.set_val(base_name)
            
            file_ext = data_files.lower().split('.')[-1]
            self.data = None

            # --- 1. MAT 파일 처리 (v7.3 호환) ---
            if file_ext == 'mat':
                try:
                    # 일반 MAT(v7 이하) 시도
                    mat_data = sio.loadmat(data_files)
                    print("Loaded as standard MAT file")
                    
                    if 'image' in mat_data:
                        self.data = mat_data['image']
                    else:
                        valid_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                        for key in valid_keys:
                            if isinstance(mat_data[key], np.ndarray) and len(mat_data[key].shape) == 3:
                                self.data = mat_data[key]
                                break
                        if self.data is None and valid_keys:
                            self.data = mat_data[valid_keys[0]]

                except (NotImplementedError, ValueError):
                    # v7.3(HDF5 형식) 대용량 파일 처리
                    print("Detected MATLAB v7.3 (HDF5). Loading with h5py...")
                    with h5py.File(data_files, 'r') as f:
                        target_key = 'image' if 'image' in f else None
                        if not target_key:
                            # 3차원 데이터셋 자동 탐색
                            for key in f.keys():
                                if isinstance(f[key], h5py.Dataset) and len(f[key].shape) == 3:
                                    target_key = key
                                    break
                        
                        if target_key:
                            self.data = f[target_key][:]
                            # v7.3 HDF5는 축 순서가 뒤집히는 경우가 많음 (B, W, H) -> (H, W, B)
                            self.data = np.transpose(self.data, (2, 1, 0))
                            print(f"v7.3 data loaded from '{target_key}' and transposed.")
                        else:
                            raise ValueError("No valid 3D dataset found in v7.3 MAT file")

            # --- 2. H5 / HDF5 파일 처리 ---
            elif file_ext in ['h5', 'hdf5']:
                with h5py.File(data_files, 'r') as f:
                    target_key = 'image' if 'image' in f else None
                    if not target_key:
                        for key in f.keys():
                            if isinstance(f[key], h5py.Dataset) and len(f[key].shape) == 3:
                                target_key = key
                                break
                    if target_key:
                        self.data = f[target_key][:]
                        self.data = np.transpose(self.data, (2, 1, 0))
                    else:
                        raise ValueError("No valid 3D dataset found")

            # --- 3. TIFF 파일 처리 ---
            elif file_ext in ['tiff', 'tif']:
                self.data = tiff.imread(data_files)
                if len(self.data.shape) == 3 and self.data.shape[0] < self.data.shape[2]:
                    self.data = np.transpose(self.data, (1, 2, 0))
                elif len(self.data.shape) == 2:
                    self.data = np.expand_dims(self.data, axis=2)

            # 데이터 로드 후 공통 처리
            if self.data is not None:
                self.data = self.data.astype(np.float32)
                
                # 기존 마커 및 스펙트럼 초기화
                self.points = []
                self.clear_markers()
                self.clear_spectra()
                
                # 슬라이더 및 이미지 업데이트
                max_band = self.data.shape[2] - 1
                self.rgb_bands = [min(b, max_band) for b in self.rgb_bands]
                
                self.update_rgb_image()
                self.ax_band.clear()
                self.band_img = self.ax_band.imshow(self.data[:, :, 0], cmap='gray')
                self.ax_band.set_title('Band 0')
                
                self.slider.valmin = 0
                self.slider.valmax = max_band
                self.slider.set_val(0)
                self.slider.ax.set_xlim(0, max_band)
                
                self.update_point_list()
                self.update_filename_display()
                self.fig.canvas.draw_idle()
                print(f"Success! Data shape: {self.data.shape}")
                
        except Exception as e:
            print(f"Error loading image: {e}")
            import traceback
            traceback.print_exc()
    
    def update_rgb_bands(self, idx, text):
        """RGB 밴드 업데이트"""
        if self.data is None:
            print("Please load an image first")
            return
        
        try:
            band = int(text)
            if 0 <= band < self.data.shape[2]:
                self.rgb_bands[idx] = band
                self.update_rgb_image()
            else:
                print(f"Band number must be between 0 and {self.data.shape[2]-1}")
                if idx == 0:
                    self.tb_r.set_val(str(self.rgb_bands[0]))
                elif idx == 1:
                    self.tb_g.set_val(str(self.rgb_bands[1]))
                else:
                    self.tb_b.set_val(str(self.rgb_bands[2]))
        except ValueError:
            print("Invalid band number")
    
    def save_falsecolor_image(self, event):
        """Falsecolor 이미지를 PNG 파일로 저장"""
        if self.data is None:
            print("No image loaded to save")
            return
        
        try:
            # 원본 코드와 동일한 방식으로 RGB 이미지 생성
            rgb = self.data[:, :, self.rgb_bands].astype(np.float32)

            # ✅ 방법 1: MATLAB 스타일 - 채널별 개별 정규화
            rgb_norm = np.zeros_like(rgb)
            for i in range(3):  # R, G, B 각 채널
                channel = rgb[:, :, i]
                low_val = np.percentile(channel, 0.1)
                high_val = np.percentile(channel, 99.9)
                
                # 클리핑 후 정규화
                channel_clipped = np.clip(channel, low_val, high_val)
                rgb_norm[:, :, i] = (channel_clipped - low_val) / (high_val - low_val + 1e-8)

            # ✅ 0-1 범위 강제 및 NaN 제거
            rgb_norm = np.clip(rgb_norm, 0, 1)  # 0-1 범위 강제
            rgb_norm = np.nan_to_num(rgb_norm)  # NaN 제거
            rgb_image = (rgb_norm * 255).astype(np.uint8)

            # 파일명 생성
            base_name = self.export_filename if self.export_filename != 'Name' else 'falsecolor'
            r_band, g_band, b_band = self.rgb_bands
            filename = f'{base_name}_falsecolor_R{r_band}_G{g_band}_B{b_band}.png'
            
            # PIL을 사용하여 이미지 저장
            pil_image = Image.fromarray(rgb_image)
            pil_image.save(filename)
            
            print(f"False color image saved as: {filename}")
            print(f"RGB bands used: R={r_band}, G={g_band}, B={b_band}")
            
            # ✅ 추가: 각 채널별 정규화 정보 출력
            print(f"Channel normalization ranges:")
            for i, channel_name in enumerate(['R', 'G', 'B']):
                channel = rgb[:, :, i]
                low_val = np.percentile(channel, 0.1)
                high_val = np.percentile(channel, 99.9)
                print(f"  {channel_name}-channel: [{low_val:.4f}, {high_val:.4f}]")
            
        except Exception as e:
            print(f"Error saving false color image: {str(e)}")


    def update_rgb_image(self):
        """RGB 이미지 업데이트"""
        if self.data is None:
            return
    
        self.ax_img.clear()
        self.markers = []    # 마커 리스트 초기화
        self.rectangles = [] # rectangles도 초기화      
        self.ax_img.axis('on')

        rgb = self.data[:, :, self.rgb_bands].astype(np.float32)

   
        rgb_norm = np.zeros_like(rgb)
        for i in range(3):  # R, G, B 각 채널
            channel = rgb[:, :, i]
            low_val = np.percentile(channel, 0.1)
            high_val = np.percentile(channel, 99.9)
            
            # 클리핑 후 정규화
            channel_clipped = np.clip(channel, low_val, high_val)
            rgb_norm[:, :, i] = (channel_clipped - low_val) / (high_val - low_val + 1e-8)
        
        self.im = self.ax_img.imshow(rgb_norm)
        self.ax_img.set_title(f'File : {self.loaded_filename}' if self.loaded_filename else 'No image loaded')

        self.plot_all_spectra()

    def get_binned_spectrum(self, x, y):
        """비닝 적용하여 스펙트럼 추출"""
        if self.data is None:
            return None, None
            
        half_bin = self.bin_size // 2
        y_start = max(0, y - half_bin)
        y_end = min(self.data.shape[0], y + half_bin + 1)
        x_start = max(0, x - half_bin)
        x_end = min(self.data.shape[1], x + half_bin + 1)
        
        region = self.data[y_start:y_end, x_start:x_end, :]
        spectrum = np.mean(region, axis=(0, 1))
        
        # SNV 정규화 적용 (활성화된 경우)
        if self.apply_snv:
            spectrum = self.apply_snv_normalization(spectrum)
        
        return spectrum, (y_start, y_end, x_start, x_end)
    
    def onclick(self, event):
        if self.data is None:
            print("Please load an image first")
            return
        
        # ✅ 이미지 클릭 처리
        if event.inaxes == self.ax_img and event.xdata and event.ydata:
            x, y = int(event.xdata), int(event.ydata)
            
            if 0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]:
                if event.button == 1:  # 좌클릭
                    # 기존 포인트 근처인지 확인 (드래그 시작)
                    if self.points:
                        distances = [(i, np.sqrt((x - px)**2 + (y - py)**2)) 
                                for i, (px, py, _) in enumerate(self.points)]
                        nearest_idx, nearest_dist = min(distances, key=lambda item: item[1])
                        
                        # 10픽셀 이내면 드래그 시작
                        if nearest_dist < 10:
                            self.dragging = True
                            self.drag_point_idx = nearest_idx
                            return
                    
                    # 새 포인트 추가
                    label = f'P{len(self.points) + 1}'
                    self.points.append((x, y, label))
                    self.plot_all_spectra()
                    self.update_point_list()
                    
                elif event.button == 3:  # 우클릭 - 포인트 삭제
                    if self.points:
                        distances = [(i, np.sqrt((x - px)**2 + (y - py)**2)) 
                                for i, (px, py, _) in enumerate(self.points)]
                        nearest_idx = min(distances, key=lambda item: item[1])[0]
                        self.points.pop(nearest_idx)
                        self.plot_all_spectra()
                        self.update_point_list()
        
        # ✅ 스펙트럼 클릭 처리 추가
        elif event.inaxes == self.ax_spec and event.xdata is not None:
            band = event.xdata
            
            # 가장 가까운 RGB 커서 찾기
            distances = [abs(band - self.rgb_bands[i]) for i in range(3)]
            nearest_idx = distances.index(min(distances))
            
            # 10 밴드 이내면 드래그 시작
            if distances[nearest_idx] < 10:
                self.dragging_cursor = nearest_idx
                print(f"Started dragging {['R', 'G', 'B'][nearest_idx]} cursor")
    


    def on_scroll(self, event):
        if event.inaxes == self.ax_img and self.data is not None:
            if event.xdata is None or event.ydata is None:
                return
                
            zoom_factor = 1.2 if event.button == 'up' else 1/1.2
            
            # 현재 축 범위
            xlim = self.ax_img.get_xlim()
            ylim = self.ax_img.get_ylim()
            
            # 이미지 크기
            img_height, img_width = self.data.shape[:2]
            
            # 현재 폭/높이
            width = xlim[1] - xlim[0]
            height = ylim[0] - ylim[1]  # Y축이 뒤집어져 있음
            
            # 새로운 폭/높이
            new_width = width / zoom_factor
            new_height = height / zoom_factor
            
            # 최소 크기 제한 (너무 많이 확대되지 않도록)
            min_width = 10  # 최소 10픽셀 폭
            min_height = 10  # 최소 10픽셀 높이
            new_width = max(new_width, min_width)
            new_height = max(new_height, min_height)
            
            # 최대 크기 제한 (축소 제한)
            new_width = min(new_width, img_width)
            new_height = min(new_height, img_height)
            
            # 마우스 중심으로 새 범위 계산
            center_x, center_y = event.xdata, event.ydata
            
            # 새로운 범위 (X축)
            new_x0 = center_x - new_width / 2
            new_x1 = center_x + new_width / 2
            
            # 새로운 범위 (Y축 - 뒤집힌 좌표계)
            new_y0 = center_y + new_height / 2  # 위쪽
            new_y1 = center_y - new_height / 2  # 아래쪽
            
            # X축 경계 처리
            if new_x0 < 0:
                shift = -new_x0
                new_x0 = 0
                new_x1 = min(img_width, new_x1 + shift)
            elif new_x1 > img_width:
                shift = new_x1 - img_width
                new_x1 = img_width
                new_x0 = max(0, new_x0 - shift)
                
            # Y축 경계 처리 (뒤집힌 좌표계 고려)
            if new_y1 < 0:
                shift = -new_y1
                new_y1 = 0
                new_y0 = min(img_height, new_y0 + shift)
            elif new_y0 > img_height:
                shift = new_y0 - img_height
                new_y0 = img_height
                new_y1 = max(0, new_y1 - shift)
            
            # 축 설정
            self.ax_img.set_xlim(new_x0, new_x1)
            self.ax_img.set_ylim(new_y0, new_y1)  # Y축 순서 주의
            
            self.fig.canvas.draw_idle()

    def on_release(self, event):
        """마우스 버튼 릴리즈 - 드래그 종료"""
        if self.dragging:
            self.dragging = False
            self.drag_point_idx = None


    
    def clear_markers(self):
        for marker in self.markers:
            marker.remove()
        for rect in self.rectangles:
            rect.remove()
        self.markers = []
        self.rectangles = []
    
    def clear_spectra(self):
        for line in self.spec_lines:
            line.remove()
        self.spec_lines = []
        if self.ax_spec.get_legend():
            self.ax_spec.get_legend().remove()
    
    def plot_all_spectra(self):
        if self.data is None:
            return
            
        self.clear_spectra()
        self.clear_markers()
        
        self.ax_spec.set_xlabel('Band Number')
        self.ax_spec.set_ylabel('Intensity' + (' (SNV Normalized)' if self.apply_snv else ''))
        title = f'Multi-point Spectra (Bin: {self.bin_size}x{self.bin_size})'
        if self.apply_snv:
            title += ' - SNV ON'
        self.ax_spec.set_title(title)
        self.ax_spec.grid(True)
        
        for i, (x, y, label) in enumerate(self.points):
            color = self.colors[i % len(self.colors)]
            spectrum, region_info = self.get_binned_spectrum(x, y)
            
            if spectrum is None:
                continue
            
            y_start, y_end, x_start, x_end = region_info
            
            line, = self.ax_spec.plot(range(len(spectrum)), spectrum, 
                            color=color, label=f'{label} ({x},{y})', linewidth=2)
            self.spec_lines.append(line)
            
            marker, = self.ax_img.plot(x, y, '+', color=color, markersize=15, markeredgewidth=2)
            self.markers.append(marker)
            
            if self.bin_size > 1:
                rect = plt.Rectangle((x_start-0.5, y_start-0.5), 
                                    x_end-x_start, y_end-y_start,
                                    fill=False, edgecolor=color, linewidth=1.5, linestyle='--')
                self.ax_img.add_patch(rect)
                self.rectangles.append(rect)
        
        if self.points:
            self.ax_spec.legend(loc='best', fontsize=8)
            
            # Autoscale 적용
            if self.autoscale:
                self.ax_spec.relim()
                self.ax_spec.autoscale_view()
            else:
                # Autoscale OFF일 때는 현재 범위 유지 (처음 설정된 범위)
                pass
        if self.points:  # 스펙트럼이 그려진 후에
            self.update_rgb_cursors()
        
        self.fig.canvas.draw_idle()

    
    def update_point_list(self):
        if self.data is None:
            text = "No image loaded\n\n"
            text += "Click 'Load Image' to start"
        elif self.points:
            text = "Selected Points:\n" + "="*30 + "\n"
            for i, (x, y, label) in enumerate(self.points):
                text += f"{label}: (x={x:3d}, y={y:3d})\n"
            if self.apply_snv:
                text += "\n[SNV: ON]"
        else:
            text = "No points selected\n\n"
            text += "Instructions:\n"
            text += "- Left click: Add point\n"
            text += "- Drag point: Move point\n"

            text += "- Right click: Remove nearest"
        
        self.point_text.set_text(text)
        self.fig.canvas.draw_idle()
    
    def update_band(self, val):
        if self.data is None:
            return
        band = int(self.slider.val)
        band_data = self.data[:, :, band].astype(np.float32)
        self.band_img.set_data(band_data)
        self.band_img.set_clim(band_data.min(), band_data.max())
        self.ax_band.set_title(f'Band {band}')
        self.fig.canvas.draw_idle()
    
    def update_bin_size(self, text):
        try:
            new_size = int(text)
            if new_size > 0 and new_size < 50:
                self.bin_size = new_size
                if self.points:
                    self.plot_all_spectra()
            else:
                print("Bin size must be between 1 and 50")
                self.bin_textbox.set_val(str(self.bin_size))
        except ValueError:
            print("Invalid bin size")
            self.bin_textbox.set_val(str(self.bin_size))
    
    def update_filename(self, text):
        """파일명 업데이트"""
        if text.endswith('.txt'):
            text = text[:-4]
        self.export_filename = text
        print(f"Export filename set to: {self.export_filename}.txt")
    
    def clear_all(self, event):
        self.points = []
        self.plot_all_spectra()
        self.update_point_list()
    
    def export_spectra(self, event):
        if not self.points:
            print("No points to export")
            return
        
        all_spectra = []
        
        for x, y, label in self.points:
            spectrum, _ = self.get_binned_spectrum(x, y)
            all_spectra.append(spectrum)
        
        spectra_array = np.column_stack(all_spectra)
        band_numbers = np.arange(len(all_spectra[0])).reshape(-1, 1)
        
        output_data = np.hstack([band_numbers, spectra_array])
        
        filename = f'{self.export_filename}.txt'
        if self.apply_snv:
            filename = f'{self.export_filename}_SNV.txt'
        
        np.savetxt(filename, output_data, fmt='%.6f', delimiter='\t')
        
        print(f"Spectra exported to {filename}")
        if self.apply_snv:
            print("(SNV normalization applied)")
        print(f"Format: Band_Number | {' | '.join([label for _, _, label in self.points])}")
    
    def show(self):
        plt.tight_layout()
        plt.show()

# 뷰어 실행
viewer = SpectrumViewer()
viewer.show()