import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QScrollArea, QTextEdit, QProgressBar, QSplitter,
    QLineEdit, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


# ═══════════════════════════════════════════════════════════════
#  PatchDataset — augmentation 지원 (JJI 스타일)
# ═══════════════════════════════════════════════════════════════
class _PatchDataset:
    """매 epoch 다른 augmentation 적용을 위한 Dataset"""
    def __init__(self, X, y, augment=False):
        import torch
        self.X = X            # (N, P, P, C) numpy
        self.y = y            # (N,) numpy
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        import numpy as np, torch
        patch = self.X[idx].copy()    # (P, P, C)

        if self.augment:
            # 공간 augmentation
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=0).copy()
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=1).copy()
            k = np.random.randint(0, 4)
            patch = np.rot90(patch, k, axes=(0, 1)).copy()

            # 스펙트럼 augmentation
            noise = np.random.normal(0, 0.02, patch.shape).astype(np.float32)
            patch = patch + noise
            scale = np.random.uniform(0.9, 1.1, (1, 1, patch.shape[2])).astype(np.float32)
            patch = patch * scale

        patch = np.ascontiguousarray(patch)
        # (P, P, C) → (C, P, P)
        x = torch.from_numpy(patch).permute(2, 0, 1)
        return x, torch.tensor(self.y[idx], dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════
#  DenseDataset — 이미지 + 농도맵 페어 + 동시 augmentation
# ═══════════════════════════════════════════════════════════════
class _DenseDataset:
    """전체 이미지 + 농도맵을 같이 augment"""
    def __init__(self, X, Y, augment=False):
        # X: (N, C, H, W)  Y: (N, 1, H, W)
        import torch
        self.X = X
        self.Y = Y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        import numpy as np, torch
        x = self.X[idx].clone() if hasattr(self.X[idx], 'clone') else torch.from_numpy(self.X[idx]).clone()
        y = self.Y[idx].clone() if hasattr(self.Y[idx], 'clone') else torch.from_numpy(self.Y[idx]).clone()
        # x: (C, H, W), y: (1, H, W)

        if self.augment:
            # 공간 augmentation — x와 y 동시 적용
            # 수평 flip
            if np.random.rand() > 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[2])
            # 수직 flip
            if np.random.rand() > 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[1])
            # 0 또는 180도만 회전 (비정사각형 이미지 호환)
            # 90/270은 H, W가 바뀌어서 배치 stack 실패
            if np.random.rand() > 0.5:
                x = torch.rot90(x, k=2, dims=[1, 2])
                y = torch.rot90(y, k=2, dims=[1, 2])

            # 스펙트럼 augmentation (x만 적용, y는 그대로)
            # Gaussian noise
            noise = torch.randn_like(x) * 0.02
            x = x + noise
            # Channel-wise scaling
            C = x.shape[0]
            scale = torch.empty(C, 1, 1).uniform_(0.9, 1.1)
            x = x * scale

        return x.float(), y.float()


# ═══════════════════════════════════════════════════════════════
#  NoAugmentWrapper — Val 데이터에서 augmentation만 끄기
# ═══════════════════════════════════════════════════════════════
class _NoAugmentWrapper:
    """원본 Dataset의 augmentation을 강제로 끔.
    Subset과 함께 사용 (Subset(NoAugWrapper(dataset), val_indices))"""
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # augmentation 잠시 끄고 가져오기
        if hasattr(self.base, 'augment'):
            _bak = self.base.augment
            self.base.augment = False
            item = self.base[idx]
            self.base.augment = _bak
            return item
        return self.base[idx]


# ═══════════════════════════════════════════════════════════════
#  간단한 2D CNN (Patch용) — JJI 스타일 (BN + 1x1 conv + Dropout)
# ═══════════════════════════════════════════════════════════════
def _make_simple_cnn(in_channels, patch_size):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=1),       # 1x1 = 채널 통합 (스펙트럼 학습)
        nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(128 * patch_size * patch_size, 256),
        nn.ReLU(inplace=True), nn.Dropout(0.2),
        nn.Linear(256, 1),
    )


def _make_1d_cnn(in_channels):
    """1D CNN — 스펙트럼만 사용 (공간정보 무시)"""
    import torch.nn as nn
    # 입력: (B, C) → (B, 1, C)로 변환 후 1D conv
    class SpectralCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Linear(32, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 1),
            )
        def forward(self, x):
            # x: (B, C) → (B, 1, C)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            elif x.dim() == 4:
                # patch (B, C, H, W) → 중심 픽셀 (B, C)
                B, C, H, W = x.shape
                cx = x[:, :, H//2, W//2]  # 중심
                x = cx.unsqueeze(1)
            return self.net(x)
    return SpectralCNN()


def _make_mlp(in_channels):
    """MLP — 스펙트럼만"""
    import torch.nn as nn
    class SpectralMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_channels, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 32),          nn.ReLU(inplace=True),
                nn.Linear(32, 1),
            )
        def forward(self, x):
            if x.dim() == 4:
                B, C, H, W = x.shape
                x = x[:, :, H//2, W//2]
            return self.net(x)
    return SpectralMLP()


# ═══════════════════════════════════════════════════════════════
#  TrainWorker
# ═══════════════════════════════════════════════════════════════
class TrainWorker(QThread):
    epoch_done  = pyqtSignal(int, float, float, float, float)  # ep, train_loss, val_loss, train_r2, val_r2
    log_msg     = pyqtSignal(str)
    fold_start  = pyqtSignal(int, int)
    finished    = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._stop  = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self._run_train()
        except Exception as e:
            import traceback
            self.log_msg.emit(f"[오류] {e}\n{traceback.format_exc()}")
        finally:
            self.finished.emit()

    # ──────────────────────────────────────────────────────────
    #  모델 빌드
    # ──────────────────────────────────────────────────────────
    def _build_model(self, mode, model_name, in_ch, encoder, pretrained, activation):
        import torch.nn as nn

        # Patch 모드
        if mode == "patch":
            patch_size = self.config.get("patch_size", 9)
            if model_name == "CNN-2D-Simple":
                return _make_simple_cnn(in_ch, patch_size), "patch"
            elif model_name.startswith("CNN-2D"):
                # ResNet/EfficientNet 백본 (torchvision)
                import torchvision.models as tvm
                weights = None
                name_map = {
                    "CNN-2D-ResNet18":        "resnet18",
                    "CNN-2D-ResNet34":        "resnet34",
                    "CNN-2D-ResNet50":        "resnet50",
                    "CNN-2D-EfficientNet-B0": "efficientnet_b0",
                    "CNN-2D-EfficientNet-B2": "efficientnet_b2",
                }
                tv_name = name_map.get(model_name, "resnet34")
                model = getattr(tvm, tv_name)(weights=None)

                # 첫 conv를 in_channels에 맞춰 교체
                if "resnet" in tv_name:
                    model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    model.fc = nn.Linear(model.fc.in_features, 1)
                elif "efficientnet" in tv_name:
                    model.features[0][0] = nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False)
                    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
                return model, "patch"
            elif model_name == "CNN-1D-Spectral":
                return _make_1d_cnn(in_ch), "spectral"
            elif model_name == "MLP-Spectral":
                return _make_mlp(in_ch), "spectral"

        # Dense 모드 (SMP U-Net 계열)
        else:
            import segmentation_models_pytorch as smp
            SMP_MAP = {
                "UNet": "Unet", "UNet++": "UnetPlusPlus",
                "MAnet": "MAnet", "FPN": "FPN",
                "Linknet": "Linknet", "PAN": "PAN",
                "DeepLabV3+": "DeepLabV3Plus",
            }
            enc_w = "imagenet" if pretrained == "imagenet" else None
            cls   = SMP_MAP.get(model_name, "Unet")
            # 활성화 함수 매핑
            act_map = {
                "없음 (linear)": None,
                "sigmoid": "sigmoid",
                "tanh": "tanh",
                "ReLU": "identity",  # ReLU는 SMP 기본 지원X, 모델 외부에서 처리
            }
            smp_act = act_map.get(activation, None)
            model = getattr(smp, cls)(
                encoder_name=encoder,
                encoder_weights=enc_w,
                in_channels=in_ch,
                classes=1,              # 회귀 → 출력 1
                activation=smp_act,
            )
            return model, "dense"

    # ──────────────────────────────────────────────────────────
    #  데이터셋 빌드
    # ──────────────────────────────────────────────────────────
    def _load_image(self, path):
        import numpy as np
        from PIL import Image
        import scipy.io as sio
        ext = path.rsplit(".", 1)[-1].lower()
        if ext == "mat":
            d = sio.loadmat(path)
            return np.array(list(d.values())[-1]).astype(np.float32)
        elif ext == "npy":
            return np.load(path).astype(np.float32)
        else:
            return np.array(Image.open(path)).astype(np.float32)

    def _load_label(self, path):
        import numpy as np
        from PIL import Image
        ext = path.rsplit(".", 1)[-1].lower()
        if ext == "mat":
            import scipy.io as sio
            d = sio.loadmat(path)
            return np.array(list(d.values())[-1])
        elif ext == "npy":
            return np.load(path)
        else:
            return np.array(Image.open(path))

    def _label_to_concentration(self, lbl):
        """라벨 정수 → 농도값 변환. 배경 픽셀은 -1 (학습 제외 표시)."""
        import numpy as np
        cfg     = self.config
        bg      = cfg.get("bg_value", 0)
        offset  = cfg.get("label_offset", -1.0)
        divisor = cfg.get("label_divisor", 10.0)
        if divisor == 0: divisor = 1.0
        conc = (lbl.astype(np.float32) + offset) / divisor
        conc[lbl == bg] = -1.0   # 배경은 -1로 표시 → 손실에서 제외
        return conc

    @staticmethod
    def _apply_snv(cube):
        """SNV (Standard Normal Variate) — 픽셀별 스펙트럼 정규화"""
        import numpy as np
        mean = np.mean(cube, axis=-1, keepdims=True)
        std  = np.std(cube,  axis=-1, keepdims=True)
        return (cube - mean) / (std + 1e-8)

    @staticmethod
    def _apply_minmax(cube):
        """Min-Max — 채널별로 [0, 1] 매핑 (이미지 단위)"""
        import numpy as np
        # axis=(0,1)이면 각 채널마다 H,W 전체에서 min/max
        ax = tuple(range(cube.ndim - 1))
        cmin = cube.min(axis=ax, keepdims=True)
        cmax = cube.max(axis=ax, keepdims=True)
        return (cube - cmin) / (cmax - cmin + 1e-8)

    @staticmethod
    def _apply_zscore(cube):
        """Z-score — 채널별 평균0/std1 (이미지 단위)"""
        import numpy as np
        ax = tuple(range(cube.ndim - 1))
        cmean = cube.mean(axis=ax, keepdims=True)
        cstd  = cube.std(axis=ax, keepdims=True)
        return (cube - cmean) / (cstd + 1e-8)

    def _normalize(self, cube, norm_type):
        """정규화 통합 함수"""
        if norm_type == "snv":
            return self._apply_snv(cube)
        elif norm_type == "minmax":
            return self._apply_minmax(cube)
        elif norm_type == "zscore":
            return self._apply_zscore(cube)
        else:
            return cube  # "none"

    def _build_patch_dataset(self, img_dir=None, lbl_dir=None):
        """Patch 모드: 라벨!=배경인 픽셀에서 패치 추출
        SNV 적용 옵션 + 파일당 최대 패치 수 제한
        """
        import numpy as np, torch, glob
        from torch.utils.data import TensorDataset

        cfg = self.config
        if img_dir is None: img_dir = cfg.get("train_img_dir","")
        if lbl_dir is None: lbl_dir = cfg.get("train_lbl_dir","")
        img_ext  = cfg.get("img_ext","mat")
        lbl_ext  = cfg.get("lbl_ext","png")
        patch    = cfg.get("patch_size", 9)
        max_pp   = cfg.get("max_patches_per_file", 10000)
        # 정규화 타입: norm_type 우선, 없으면 use_snv → snv/none
        norm_type = cfg.get("norm_type")
        if norm_type is None:
            norm_type = "snv" if cfg.get("use_snv", False) else "none"
        half     = patch // 2

        img_files = sorted(glob.glob(os.path.join(img_dir, f"*.{img_ext}")))
        lbl_files = sorted(glob.glob(os.path.join(lbl_dir, f"*.{lbl_ext}")))
        if not img_files:
            raise FileNotFoundError(f"이미지 없음: {img_dir}/*.{img_ext}")

        self.log_msg.emit(f"  이미지 {len(img_files)}개  ·  patch={patch}  max/file={max_pp:,}  norm={norm_type}")

        patches, targets = [], []
        rng = np.random.default_rng(cfg.get("seed", 42))
        for ip, lp in zip(img_files, lbl_files):
            img = self._load_image(ip)
            lbl = self._load_label(lp)
            conc = self._label_to_concentration(lbl)
            # (H, W, C) 표준화
            if img.ndim == 2:
                img = img[..., None]
            elif img.shape[0] < img.shape[-1]:
                # (C, H, W) → (H, W, C)
                img = np.transpose(img, (1, 2, 0))

            # 정규화 (선택)
            img = self._normalize(img, norm_type).astype(np.float32)

            # padding (edge 모드, JJI 방식)
            padded = np.pad(img, ((half, half), (half, half), (0, 0)), mode='edge')

            # 유효 픽셀 좌표
            ys, xs = np.where(conc >= 0)
            n = len(ys)
            if n == 0:
                continue
            # 파일당 최대 패치 수 랜덤 샘플링
            if n > max_pp:
                sel = rng.choice(n, max_pp, replace=False)
                ys = ys[sel]; xs = xs[sel]

            for y, x in zip(ys, xs):
                # (P, P, C)로 저장 → __getitem__에서 augment 후 transpose
                patches.append(padded[y:y+patch, x:x+patch, :])
                targets.append(conc[y, x])

        self.log_msg.emit(f"    → 총 패치 수: {len(patches):,}개")
        if len(patches) == 0:
            raise RuntimeError("학습 가능한 패치가 없습니다.")

        # (N, P, P, C)로 저장 (augmentation 위해)
        X = np.stack(patches).astype(np.float32)
        y = np.array(targets, dtype=np.float32)

        # Augmentation 사용 여부에 따라 다른 Dataset 클래스 사용
        if cfg.get("use_augmentation", False):
            return _PatchDataset(X, y, augment=True)
        else:
            # 단순 TensorDataset (이미 (C,P,P) 변환)
            Xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
            yt = torch.from_numpy(y)
            return TensorDataset(Xt, yt)

    def _build_dense_dataset(self, img_dir=None, lbl_dir=None):
        """Dense 모드: 전체 이미지 + 농도맵 페어"""
        import numpy as np, torch, glob
        from torch.utils.data import TensorDataset
        try:
            from utils.pad_utils import get_pad_multiple, pad_np
            multiple = get_pad_multiple(self.config.get("model_name","UNet"))
        except Exception:
            multiple = 32
            def pad_np(arr, multiple=32, is_label=False, fill_value=0):
                h, w = arr.shape[:2]
                ph = (multiple - h % multiple) % multiple
                pw = (multiple - w % multiple) % multiple
                if ph == 0 and pw == 0: return arr, h, w
                pw_arr = ((0,ph),(0,pw),(0,0)) if arr.ndim == 3 else ((0,ph),(0,pw))
                return np.pad(arr, pw_arr, mode='reflect' if not is_label else 'constant',
                              constant_values=fill_value if is_label else 0), h, w

        cfg = self.config
        if img_dir is None: img_dir = cfg.get("train_img_dir","")
        if lbl_dir is None: lbl_dir = cfg.get("train_lbl_dir","")
        img_ext = cfg.get("img_ext","mat")
        lbl_ext = cfg.get("lbl_ext","png")

        img_files = sorted(glob.glob(os.path.join(img_dir, f"*.{img_ext}")))
        lbl_files = sorted(glob.glob(os.path.join(lbl_dir, f"*.{lbl_ext}")))
        if not img_files:
            raise FileNotFoundError(f"이미지 없음: {img_dir}/*.{img_ext}")

        use_snv = cfg.get("use_snv", False)
        norm_type = cfg.get("norm_type")
        if norm_type is None:
            norm_type = "snv" if use_snv else "none"
        self.log_msg.emit(f"  이미지 {len(img_files)}개 로딩  ·  norm={norm_type}")
        imgs, concs = [], []
        orig_h = orig_w = None
        for ip, lp in zip(img_files, lbl_files):
            img = self._load_image(ip)
            lbl = self._load_label(lp)
            conc = self._label_to_concentration(lbl)
            # (H, W, C) 표준화
            if img.ndim == 2:
                img = img[..., None]
            elif img.shape[0] < img.shape[-1]:
                img = np.transpose(img, (1, 2, 0))
            # 정규화
            img = self._normalize(img, norm_type).astype(np.float32)
            img_p, oh, ow = pad_np(img, multiple=multiple, is_label=False)
            conc_p, _, _ = pad_np(conc, multiple=multiple, is_label=True, fill_value=-1.0)
            imgs.append(img_p); concs.append(conc_p)
            if orig_h is None: orig_h, orig_w = oh, ow

        if orig_h:
            self.log_msg.emit(f"    자동 패딩: {orig_h}x{orig_w} → "
                              f"{imgs[0].shape[0]}x{imgs[0].shape[1]} (multiple={multiple})")

        X = np.stack(imgs)
        Y = np.stack(concs)
        Xt = torch.from_numpy(X).float()
        if Xt.ndim == 4 and Xt.shape[-1] < Xt.shape[-2]:
            Xt = Xt.permute(0, 3, 1, 2)
        elif Xt.ndim == 3:
            Xt = Xt.unsqueeze(1)
        Yt = torch.from_numpy(Y).float().unsqueeze(1)
        self.config["_pad_multiple"] = multiple
        self.config["_orig_size"]    = [orig_h, orig_w]

        # Augmentation 옵션에 따라 다른 Dataset 클래스 사용
        if cfg.get("use_augmentation", False):
            self.log_msg.emit(f"  Augmentation: ON (flip/rotate + noise + spectral scaling)")
            return _DenseDataset(Xt, Yt, augment=True)
        else:
            return TensorDataset(Xt, Yt)

    def _build_loaders(self, dataset, val_dataset=None):
        """K-Fold / Hold-out / Manual에 맞춰 DataLoader 리스트 반환
        manual 모드일 때는 val_dataset이 별도로 전달됨"""
        from torch.utils.data import DataLoader, random_split, Subset
        cfg   = self.config
        batch = cfg.get("batch", 16)
        split = cfg.get("split_mode", "manual")
        # Val용 wrapper (augmentation 강제 OFF)
        val_view = _NoAugmentWrapper(dataset)

        if split == "manual" and val_dataset is not None:
            self.log_msg.emit(f"  Manual: Train {len(dataset)} / Val {len(val_dataset)}")
            return [(DataLoader(dataset,     batch_size=batch, shuffle=True,
                                num_workers=0, pin_memory=True),
                     DataLoader(val_dataset, batch_size=batch, shuffle=False,
                                num_workers=0, pin_memory=True))]
        elif split == "holdout":
            ratio = cfg.get("holdout_ratio", 0.2)
            n_v   = max(1, int(len(dataset) * ratio))
            n_t   = len(dataset) - n_v
            # 인덱스 분할 (augmentation은 Subset이 아닌 원본에서 결정됨)
            import torch
            g = torch.Generator().manual_seed(cfg.get("seed", 42))
            indices = torch.randperm(len(dataset), generator=g).tolist()
            tr_idx = indices[:n_t]; vl_idx = indices[n_t:]
            self.log_msg.emit(f"  Hold-out: Train {n_t} / Val {n_v}")
            return [(DataLoader(Subset(dataset, tr_idx), batch_size=batch, shuffle=True,
                                num_workers=0, pin_memory=True),
                     DataLoader(Subset(val_view, vl_idx), batch_size=batch, shuffle=False,
                                num_workers=0, pin_memory=True))]
        elif split == "kfold":
            from sklearn.model_selection import KFold
            k = cfg.get("kfold_k", 5)
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            X = list(range(len(dataset)))
            folds = []
            for tr_idx, vl_idx in kf.split(X):
                folds.append((
                    DataLoader(Subset(dataset, tr_idx), batch_size=batch, shuffle=True,
                               num_workers=0, pin_memory=True),
                    DataLoader(Subset(val_view, vl_idx), batch_size=batch, shuffle=False,
                               num_workers=0, pin_memory=True),
                ))
            self.log_msg.emit(f"  K-Fold: K={k}")
            return folds
        else:
            # manual인데 val 폴더 없는 경우 → fallback
            self.log_msg.emit("  [경고] Manual인데 Val 폴더 비어있어 20% hold-out 적용")
            ratio = 0.2
            n_v   = max(1, int(len(dataset) * ratio))
            n_t   = len(dataset) - n_v
            import torch
            g = torch.Generator().manual_seed(cfg.get("seed", 42))
            indices = torch.randperm(len(dataset), generator=g).tolist()
            tr_idx = indices[:n_t]; vl_idx = indices[n_t:]
            return [(DataLoader(Subset(dataset, tr_idx), batch_size=batch, shuffle=True,
                                num_workers=0, pin_memory=True),
                     DataLoader(Subset(val_view, vl_idx), batch_size=batch, shuffle=False,
                                num_workers=0, pin_memory=True))]

    # ──────────────────────────────────────────────────────────
    #  손실 + 지표
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _masked_loss(pred, target, loss_type="MSE"):
        """배경(-1)은 제외하고 손실 계산"""
        import torch
        import torch.nn.functional as F
        mask = (target >= 0)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        p = pred[mask]
        t = target[mask]
        if loss_type == "MSE":
            return F.mse_loss(p, t)
        elif loss_type == "L1":
            return F.l1_loss(p, t)
        elif loss_type == "Huber":
            return F.smooth_l1_loss(p, t)
        return F.mse_loss(p, t)

    @staticmethod
    def _r2_score(pred, target):
        """R² (배경 제외)"""
        import torch
        mask = (target >= 0)
        if mask.sum() < 2:
            return 0.0
        p = pred[mask]; t = target[mask]
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum() + 1e-9
        return float(1 - ss_res / ss_tot)

    @staticmethod
    def _r2_from_arrays(preds_list, targets_list):
        """누적된 텐서 리스트에서 한 번에 R² 계산 (배치별 평균보다 안정적)"""
        import torch
        if not preds_list:
            return 0.0
        p = torch.cat(preds_list)
        t = torch.cat(targets_list)
        if len(t) < 2:
            return 0.0
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum() + 1e-9
        return float(1 - ss_res / ss_tot)

    # ──────────────────────────────────────────────────────────
    #  학습 루프
    # ──────────────────────────────────────────────────────────
    def _run_train(self):
        import torch, os, json, numpy as np, random
        cfg      = self.config
        epochs   = cfg.get("epochs", 100)
        ckpt_dir = cfg.get("ckpt_dir", "./checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        # seed 고정 (재현성)
        seed = cfg.get("seed", 42)
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mode   = cfg.get("mode", "patch")
        self.log_msg.emit(f"[학습 시작]  디바이스: {device}")
        self.log_msg.emit(f"  모드: {mode}  |  모델: {cfg.get('model_name')}")

        # activation 체크 (Dense 회귀에서 sigmoid/tanh는 비추천)
        if mode == "dense":
            act = cfg.get("activation", "없음 (linear)")
            if act != "없음 (linear)":
                self.log_msg.emit(f"  [경고] Dense 회귀에 활성화 '{act}' 사용 중 — '없음 (linear)' 권장")

        self.log_msg.emit(f"  Epochs: {epochs}  |  LR: {cfg.get('lr')}  |  Seed: {seed}")

        # 데이터셋 구축
        split = cfg.get("split_mode", "manual")
        if mode == "patch":
            self.log_msg.emit("[Train 데이터셋 구축]")
            dataset = self._build_patch_dataset()
            val_dataset = None
            if split == "manual":
                val_img = cfg.get("val_img_dir","").strip()
                val_lbl = cfg.get("val_lbl_dir","").strip()
                if val_img and val_lbl and os.path.isdir(val_img) and os.path.isdir(val_lbl):
                    self.log_msg.emit("[Val 데이터셋 구축]")
                    # Val에는 augmentation 적용 안 함
                    _aug_backup = cfg.get("use_augmentation", False)
                    cfg["use_augmentation"] = False
                    val_dataset = self._build_patch_dataset(val_img, val_lbl)
                    cfg["use_augmentation"] = _aug_backup
        else:
            self.log_msg.emit("[Train 데이터셋 구축]")
            dataset = self._build_dense_dataset()
            val_dataset = None
            if split == "manual":
                val_img = cfg.get("val_img_dir","").strip()
                val_lbl = cfg.get("val_lbl_dir","").strip()
                if val_img and val_lbl and os.path.isdir(val_img) and os.path.isdir(val_lbl):
                    self.log_msg.emit("[Val 데이터셋 구축]")
                    _aug_backup = cfg.get("use_augmentation", False)
                    cfg["use_augmentation"] = False
                    val_dataset = self._build_dense_dataset(val_img, val_lbl)
                    cfg["use_augmentation"] = _aug_backup

        folds = self._build_loaders(dataset, val_dataset)
        is_kfold = cfg.get("split_mode") == "kfold" and len(folds) > 1
        fold_results = []

        in_ch     = cfg.get("in_channels", 6)
        encoder   = cfg.get("encoder", "resnet34")
        pre       = cfg.get("pretrained", "없음 (scratch)")
        activ     = cfg.get("activation", "없음 (linear)")
        loss_type = cfg.get("loss_type", "MSE")

        for fold_idx, (train_loader, val_loader) in enumerate(folds):
            if self._stop: break
            if len(folds) > 1:
                self.log_msg.emit(f"\n── Fold {fold_idx+1}/{len(folds)} ──")
            self.fold_start.emit(fold_idx + 1, len(folds))

            model, _ = self._build_model(mode, cfg.get("model_name"), in_ch, encoder, pre, activ)
            model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3),
                                          weight_decay=cfg.get("weight_decay", 1e-4))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10)

            best_vl = float('inf'); best_vr2 = -float('inf')
            es_counter = 0
            es_patience = cfg.get("es_patience", 30)
            use_es = cfg.get("use_early_stopping", True)

            for ep in range(1, epochs + 1):
                if self._stop:
                    self.log_msg.emit("[중단]"); break

                # ── Train ──
                model.train()
                tl_sum = 0.0; nb = 0
                train_preds, train_targets = [], []
                for X, y in train_loader:
                    X = X.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(X).squeeze(1) if mode == "dense" else model(X).squeeze(-1)
                    if mode == "dense":
                        if y.dim() == 4: y = y.squeeze(1)
                    loss = self._masked_loss(pred, y, loss_type)
                    loss.backward()
                    if cfg.get("grad_clip", True):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    tl_sum += loss.item()
                    nb += 1
                    # 누적 (배경 제외)
                    with torch.no_grad():
                        m_ = (y >= 0)
                        if m_.sum() > 0:
                            train_preds.append(pred.detach()[m_].cpu())
                            train_targets.append(y[m_].cpu())
                avg_tl  = tl_sum / max(nb, 1)
                avg_tr2 = self._r2_from_arrays(train_preds, train_targets)

                # ── Val ──
                model.eval()
                vl_sum = 0.0; nb = 0
                val_preds, val_targets = [], []
                with torch.no_grad():
                    for X, y in val_loader:
                        X = X.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        pred = model(X).squeeze(1) if mode == "dense" else model(X).squeeze(-1)
                        if mode == "dense" and y.dim() == 4:
                            y = y.squeeze(1)
                        loss = self._masked_loss(pred, y, loss_type)
                        vl_sum += loss.item()
                        nb += 1
                        m_ = (y >= 0)
                        if m_.sum() > 0:
                            val_preds.append(pred[m_].cpu())
                            val_targets.append(y[m_].cpu())
                avg_vl  = vl_sum / max(nb, 1)
                avg_vr2 = self._r2_from_arrays(val_preds, val_targets)

                scheduler.step(avg_vl)
                cur_lr = optimizer.param_groups[0]['lr']

                # best 저장
                if avg_vl < best_vl:
                    best_vl  = avg_vl
                    best_vr2 = avg_vr2
                    es_counter = 0
                    suffix = f"_fold{fold_idx+1}" if len(folds) > 1 else ""
                    best_path = os.path.join(ckpt_dir, f"best_model{suffix}.pth")
                    torch.save(model.state_dict(), best_path)

                    model_config = {
                        "mode":         mode,
                        "model_name":   cfg.get("model_name"),
                        "encoder":      encoder,
                        "in_channels":  in_ch,
                        "patch_size":   cfg.get("patch_size", 9),
                        "norm_type":    cfg.get("norm_type", "snv" if cfg.get("use_snv",False) else "none"),
                        "use_snv":      cfg.get("use_snv", False),  # 호환성
                        "activation":   activ,
                        "out_min":      cfg.get("out_min", 0.0),
                        "out_max":      cfg.get("out_max", 1.0),
                        "label_offset": cfg.get("label_offset", -1.0),
                        "label_divisor":cfg.get("label_divisor", 10.0),
                        "bg_value":     cfg.get("bg_value", 0),
                        "pad_multiple": cfg.get("_pad_multiple", 32),
                        "orig_size":    cfg.get("_orig_size", None),
                        "best_epoch":   ep,
                        "val_loss":     round(avg_vl, 6),
                        "val_r2":       round(avg_vr2, 6),
                    }
                    with open(best_path.replace(".pth","_config.json"), "w", encoding="utf-8") as f:
                        json.dump(model_config, f, indent=2, ensure_ascii=False)
                    self.log_msg.emit(f"  ✔ Best ep{ep} | val_loss={avg_vl:.5f} R²={avg_vr2:.4f}")
                else:
                    es_counter += 1

                self.epoch_done.emit(ep, avg_tl, avg_vl, avg_tr2, avg_vr2)
                self.log_msg.emit(
                    f"Ep {ep:03d}/{epochs} | Loss {avg_tl:.5f}/{avg_vl:.5f} | "
                    f"R² {avg_tr2:.4f}/{avg_vr2:.4f} | LR {cur_lr:.2e} | ES {es_counter}/{es_patience}"
                )

                if use_es and es_counter >= es_patience:
                    self.log_msg.emit(f"  Early Stopping (ep{ep})")
                    break

            fold_results.append({
                "fold": fold_idx + 1,
                "best_val_loss": round(best_vl, 6),
                "best_val_r2":   round(best_vr2, 6),
            })

        # K-Fold 결과 요약
        if is_kfold and not self._stop:
            import numpy as np
            r2_list   = [r["best_val_r2"]   for r in fold_results]
            loss_list = [r["best_val_loss"] for r in fold_results]
            summary = {
                "model":          cfg.get("model_name"),
                "mode":           mode,
                "encoder":        cfg.get("encoder"),
                "norm_type":      cfg.get("norm_type", "snv" if cfg.get("use_snv",False) else "none"),
                "use_augmentation": cfg.get("use_augmentation", False),
                "k_folds":        len(folds),
                "fold_results":   fold_results,
                "mean_val_r2":    round(float(np.mean(r2_list)),    6),
                "std_val_r2":     round(float(np.std(r2_list)),     6),
                "min_val_r2":     round(float(np.min(r2_list)),     6),
                "max_val_r2":     round(float(np.max(r2_list)),     6),
                "median_val_r2":  round(float(np.median(r2_list)),  6),
                "mean_val_loss":  round(float(np.mean(loss_list)),  6),
                "std_val_loss":   round(float(np.std(loss_list)),   6),
                "min_val_loss":   round(float(np.min(loss_list)),   6),
                "max_val_loss":   round(float(np.max(loss_list)),   6),
                "median_val_loss":round(float(np.median(loss_list)),6),
            }
            with open(os.path.join(ckpt_dir, "kfold_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            self.log_msg.emit(f"\n{'='*50}")
            self.log_msg.emit(f"[K-Fold 요약]")
            for r in fold_results:
                self.log_msg.emit(f"  Fold {r['fold']}: R²={r['best_val_r2']:.4f} loss={r['best_val_loss']:.5f}")
            self.log_msg.emit(f"  평균 R² : {summary['mean_val_r2']:.4f} ± {summary['std_val_r2']:.4f}")
            self.log_msg.emit(f"  Min~Max: {summary['min_val_r2']:.4f} ~ {summary['max_val_r2']:.4f}")
            self.log_msg.emit(f"  → kfold_summary.json 저장 완료")

            # ── Full Data 학습 (모든 fold 검증 후 최종 배포용 모델) ──
            if not self._stop:
                self.log_msg.emit(f"\n{'='*50}")
                self.log_msg.emit("[Full Data 최종 학습 시작]")
                self.log_msg.emit("  K-Fold 검증 완료 → 이제 전체 데이터로 마지막 모델 학습")

                from torch.utils.data import DataLoader
                full_loader = DataLoader(dataset, batch_size=cfg.get("batch", 16),
                                         shuffle=True, num_workers=0, pin_memory=True)
                self.log_msg.emit(f"  Full 데이터: {len(dataset):,}개 샘플")
                self.fold_start.emit(len(folds) + 1, len(folds) + 1)

                model, _ = self._build_model(mode, cfg.get("model_name"), in_ch, encoder, pre, activ)
                model = model.to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3),
                                              weight_decay=cfg.get("weight_decay", 1e-4))
                # Full 학습은 validation 없음 → CosineAnnealing으로 epoch별 부드럽게 감소
                full_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=cfg.get("lr", 1e-3) * 0.01
                )
                self.log_msg.emit(f"  Scheduler: CosineAnnealing (LR {cfg.get('lr')} → {cfg.get('lr')*0.01:.2e})")

                # Full 학습은 validation 없으니 train R² 기준으로 best 저장
                best_tr2 = -float('inf')

                for ep in range(1, epochs + 1):
                    if self._stop:
                        self.log_msg.emit("[중단]"); break
                    model.train()
                    tl_sum = 0.0; nb = 0
                    train_preds, train_targets = [], []
                    for X, y in full_loader:
                        X = X.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)
                        pred = model(X).squeeze(1) if mode == "dense" else model(X).squeeze(-1)
                        if mode == "dense" and y.dim() == 4:
                            y = y.squeeze(1)
                        loss = self._masked_loss(pred, y, loss_type)
                        loss.backward()
                        if cfg.get("grad_clip", True):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        tl_sum += loss.item(); nb += 1
                        with torch.no_grad():
                            m_ = (y >= 0)
                            if m_.sum() > 0:
                                train_preds.append(pred.detach()[m_].cpu())
                                train_targets.append(y[m_].cpu())
                    avg_tl  = tl_sum / max(nb, 1)
                    avg_tr2 = self._r2_from_arrays(train_preds, train_targets)

                    # Cosine scheduler step (epoch마다 LR 점진적 감소)
                    full_scheduler.step()
                    cur_lr = optimizer.param_groups[0]['lr']

                    # best 저장 (train R² 기준)
                    if avg_tr2 > best_tr2:
                        best_tr2 = avg_tr2
                        full_path = os.path.join(ckpt_dir, "best_model_full.pth")
                        torch.save(model.state_dict(), full_path)
                        full_config = {
                            "mode":         mode,
                            "model_name":   cfg.get("model_name"),
                            "encoder":      encoder,
                            "in_channels":  in_ch,
                            "patch_size":   cfg.get("patch_size", 9),
                            "norm_type":    cfg.get("norm_type", "snv" if cfg.get("use_snv",False) else "none"),
                            "use_snv":      cfg.get("use_snv", False),
                            "activation":   activ,
                            "out_min":      cfg.get("out_min", 0.0),
                            "out_max":      cfg.get("out_max", 1.0),
                            "label_offset": cfg.get("label_offset", -1.0),
                            "label_divisor":cfg.get("label_divisor", 10.0),
                            "bg_value":     cfg.get("bg_value", 0),
                            "pad_multiple": cfg.get("_pad_multiple", 32),
                            "orig_size":    cfg.get("_orig_size", None),
                            "fold":         "full",
                            "best_epoch":   ep,
                            "train_loss":   round(avg_tl, 6),
                            "train_r2":     round(avg_tr2, 6),
                        }
                        with open(full_path.replace(".pth","_config.json"), "w", encoding="utf-8") as f:
                            json.dump(full_config, f, indent=2, ensure_ascii=False)

                    self.epoch_done.emit(ep, avg_tl, avg_tl, avg_tr2, avg_tr2)
                    if ep % 10 == 0 or ep == 1:
                        self.log_msg.emit(
                            f"  [Full] Ep {ep:03d}/{epochs} | Loss {avg_tl:.5f} | R² {avg_tr2:.4f} | LR {cur_lr:.2e}"
                        )

                self.log_msg.emit(f"  ✔ Full 모델 저장 → best_model_full.pth (R²={best_tr2:.4f})")

        self.log_msg.emit("\n✔ 학습 완료")


# ═══════════════════════════════════════════════════════════════
#  TrainTab UI
# ═══════════════════════════════════════════════════════════════
class TrainTab(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state
        self.worker = None
        self._data_tab = None
        self.train_loss_data = []
        self.val_loss_data   = []
        self.train_r2_data   = []
        self.val_r2_data     = []
        self._build()

    def set_data_tab(self, data_tab):
        self._data_tab = data_tab

    def _get_data_config(self):
        if self.state.get("train_img_dir", "").strip():
            return {k: self.state.get(k) for k in [
                "train_img_dir","train_lbl_dir","val_img_dir","val_lbl_dir",
                "img_ext","lbl_ext","split_mode","holdout_ratio","kfold_k",
                "bg_value","label_offset","label_divisor"]}
        if self._data_tab:
            dt = self._data_tab
            mode_map = {0:"manual",1:"holdout",2:"kfold"}
            return {
                "train_img_dir": dt.train_img_edit.text().strip(),
                "train_lbl_dir": dt.train_lbl_edit.text().strip(),
                "val_img_dir":   dt.val_img_edit.text().strip(),
                "val_lbl_dir":   dt.val_lbl_edit.text().strip(),
                "img_ext":       dt.img_ext.currentText(),
                "lbl_ext":       dt.lbl_ext.currentText(),
                "split_mode":    mode_map.get(dt.split_btns.checkedId(), "manual"),
                "holdout_ratio": dt.holdout_spin.value(),
                "kfold_k":       dt.kfold_spin.value(),
                "bg_value":      dt.bg_value.value(),
                "label_offset":  dt.offset.value(),
                "label_divisor": dt.divisor.value(),
            }
        return {}

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)

        # ── 좌측 하이퍼파라미터 ───────────────────────────────
        left_w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); scroll.setWidget(inner)
        lv = QVBoxLayout(inner); lv.setSpacing(12); lv.setContentsMargins(16,16,16,16)

        hp_g = QGroupBox("하이퍼파라미터")
        hp = QGridLayout(hp_g)
        hp.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs = QSpinBox(); self.epochs.setRange(1, 10000); self.epochs.setValue(100)
        hp.addWidget(self.epochs, 0, 1)

        hp.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch = QSpinBox(); self.batch.setRange(1, 1024); self.batch.setValue(16)
        hp.addWidget(self.batch, 1, 1)

        hp.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr = QDoubleSpinBox(); self.lr.setDecimals(6); self.lr.setRange(1e-6, 1.0)
        self.lr.setValue(1e-3); self.lr.setSingleStep(1e-4)
        hp.addWidget(self.lr, 2, 1)

        lr_hint = QLabel("일반적 시작값: 1e-3  ·  발산 시 1e-4로 낮춰서 재시도")
        lr_hint.setObjectName("hint")
        hp.addWidget(lr_hint, 3, 1)

        hp.addWidget(QLabel("Weight Decay:"), 4, 0)
        self.wd = QDoubleSpinBox(); self.wd.setDecimals(6); self.wd.setRange(0, 0.1)
        self.wd.setValue(1e-4); self.wd.setSingleStep(1e-5)
        hp.addWidget(self.wd, 4, 1)

        hp.addWidget(QLabel("손실 함수:"), 5, 0)
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["MSE", "L1", "Huber"])
        hp.addWidget(self.loss_combo, 5, 1)

        hp.setColumnStretch(1, 1)
        lv.addWidget(hp_g)

        es_g = QGroupBox("Early Stopping")
        eg = QGridLayout(es_g)
        self.use_es = QCheckBox("Early Stopping 사용"); self.use_es.setChecked(True)
        eg.addWidget(self.use_es, 0, 0, 1, 2)
        eg.addWidget(QLabel("Patience:"), 1, 0)
        self.es_patience = QSpinBox(); self.es_patience.setRange(1, 1000); self.es_patience.setValue(30)
        eg.addWidget(self.es_patience, 1, 1)
        lv.addWidget(es_g)

        ck_g = QGroupBox("저장 경로")
        cg = QGridLayout(ck_g)
        self.ckpt_path_edit = QLineEdit(); self.ckpt_path_edit.setPlaceholderText("./checkpoints")
        cg.addWidget(self.ckpt_path_edit, 0, 0)
        btn_ck = QPushButton("찾기"); btn_ck.setObjectName("secondary"); btn_ck.setFixedWidth(44)
        btn_ck.clicked.connect(self._browse_ckpt)
        cg.addWidget(btn_ck, 0, 1)
        lv.addWidget(ck_g)

        self.start_btn = QPushButton("▶  학습 시작")
        self.start_btn.clicked.connect(self._start_train)
        self.stop_btn = QPushButton("■  중단"); self.stop_btn.setObjectName("secondary")
        self.stop_btn.clicked.connect(self._stop_train); self.stop_btn.setEnabled(False)
        bh = QHBoxLayout(); bh.addWidget(self.start_btn); bh.addWidget(self.stop_btn)
        lv.addLayout(bh)
        lv.addStretch()

        lw = QVBoxLayout(left_w); lw.setContentsMargins(0,0,0,0); lw.addWidget(scroll)
        splitter.addWidget(left_w)

        # ── 우측 그래프 + 로그 ─────────────────────────────────
        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setContentsMargins(16,16,16,16); rv.setSpacing(8)

        self.progress = QProgressBar(); self.progress.setValue(0)
        rv.addWidget(self.progress)

        self.epoch_label = QLabel("대기 중")
        self.epoch_label.setStyleSheet("font-weight:700; font-size:13px;")
        rv.addWidget(self.epoch_label)

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
            from matplotlib.figure import Figure
            self.fig = Figure(figsize=(7, 4), facecolor='white')
            self.ax1 = self.fig.add_subplot(121)
            self.ax2 = self.fig.add_subplot(122)
            self.canvas = Canvas(self.fig)
            rv.addWidget(self.canvas, 1)
            self.has_mpl = True
        except ImportError:
            self.has_mpl = False
            rv.addWidget(QLabel("matplotlib 없음"))

        rv.addWidget(QLabel("학습 로그"))
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(180)
        rv.addWidget(self.log_box)

        splitter.addWidget(right_w)
        splitter.setSizes([380, 720])
        outer.addWidget(splitter)

    def _browse_ckpt(self):
        path = QFileDialog.getExistingDirectory(self, "체크포인트 폴더")
        if path: self.ckpt_path_edit.setText(path)

    def _start_train(self):
        self.train_loss_data.clear(); self.val_loss_data.clear()
        self.train_r2_data.clear();   self.val_r2_data.clear()
        self.log_box.clear(); self.progress.setValue(0)

        data_cfg = self._get_data_config()
        if not data_cfg.get("train_img_dir") or not data_cfg.get("train_lbl_dir"):
            self.log_box.append("[오류] 데이터 탭에서 이미지/라벨 경로를 설정하세요"); return

        config = {
            **data_cfg,
            "mode":         self.state.get("mode", "patch"),
            "model_name":   self.state.get("model_name", "CNN-2D-Simple"),
            "encoder":      self.state.get("encoder", "resnet34"),
            "pretrained":   self.state.get("pretrained", "없음 (scratch)"),
            "in_channels":  self.state.get("in_channels", 6),
            "patch_size":   self.state.get("patch_size", 3),
            "max_patches_per_file": self.state.get("max_patches_per_file", 10000),
            "use_snv":      self.state.get("use_snv", False),
            "norm_type":    self.state.get("norm_type", "snv" if self.state.get("use_snv",False) else "none"),
            "use_augmentation": self.state.get("use_augmentation", True),
            "activation":   self.state.get("activation", "없음 (linear)"),
            "out_min":      self.state.get("out_min", 0.0),
            "out_max":      self.state.get("out_max", 1.0),
            "epochs":       self.epochs.value(),
            "batch":        self.batch.value(),
            "lr":           self.lr.value(),
            "weight_decay": self.wd.value(),
            "loss_type":    self.loss_combo.currentText(),
            "use_early_stopping": self.use_es.isChecked(),
            "es_patience":  self.es_patience.value(),
            "grad_clip":    True,
            "seed":         42,
            "ckpt_dir":     self.ckpt_path_edit.text().strip() or "./checkpoints",
        }
        self.total_epochs = config["epochs"]
        self.worker = TrainWorker(config)
        self.worker.epoch_done.connect(self._on_epoch)
        self.worker.fold_start.connect(self._on_fold_start)
        self.worker.log_msg.connect(self.log_box.append)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)

    def _stop_train(self):
        if self.worker: self.worker.stop()

    def _on_fold_start(self, fold_idx, total):
        self.train_loss_data.clear(); self.val_loss_data.clear()
        self.train_r2_data.clear();   self.val_r2_data.clear()
        self.progress.setValue(0)
        if total > 1:
            self.epoch_label.setText(f"Fold {fold_idx}/{total} 학습 중")
        if self.has_mpl:
            self.ax1.cla(); self.ax2.cla(); self.canvas.draw()

    def _on_epoch(self, ep, tl, vl, tr2, vr2):
        self.train_loss_data.append(tl); self.val_loss_data.append(vl)
        self.train_r2_data.append(tr2);  self.val_r2_data.append(vr2)
        self.progress.setValue(int(ep / self.total_epochs * 100))
        self.epoch_label.setText(
            f"Epoch {ep}/{self.total_epochs}  |  Val Loss: {vl:.5f}  R²: {vr2:.4f}"
        )
        if self.has_mpl: self._update_plot()

    def _update_plot(self):
        xs = list(range(1, len(self.train_loss_data)+1))
        self.ax1.cla(); self.ax2.cla()
        self.ax1.plot(xs, self.train_loss_data, label="Train", color="#2563eb", lw=1.5)
        self.ax1.plot(xs, self.val_loss_data,   label="Val",   color="#ef4444", lw=1.5)
        self.ax1.set_title("Loss"); self.ax1.legend(fontsize=9); self.ax1.grid(alpha=0.3)
        self.ax1.set_xlabel("Epoch")
        self.ax2.plot(xs, self.train_r2_data, label="Train", color="#2563eb", lw=1.5)
        self.ax2.plot(xs, self.val_r2_data,   label="Val",   color="#ef4444", lw=1.5)
        self.ax2.set_title("R² Score"); self.ax2.legend(fontsize=9); self.ax2.grid(alpha=0.3)
        self.ax2.set_xlabel("Epoch"); self.ax2.set_ylim(-0.1, 1.05)
        self.fig.tight_layout()
        self.canvas.draw()

    def _on_finished(self):
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.log_box.append("✔ 완료"); self.progress.setValue(100)
        self._save_train_snapshot()

    def _save_train_snapshot(self):
        import os, json, datetime
        if not self.has_mpl or not self.train_loss_data: return
        ckpt_dir = self.ckpt_path_edit.text().strip() or "./checkpoints"
        try:
            import matplotlib.pyplot as plt
            xs = list(range(1, len(self.train_loss_data)+1))
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
            a1.plot(xs, self.train_loss_data, label="Train", color="#2563eb", lw=1.5)
            a1.plot(xs, self.val_loss_data,   label="Val",   color="#ef4444", lw=1.5)
            a1.set_title("Loss"); a1.legend(); a1.grid(alpha=0.3); a1.set_xlabel("Epoch")
            a2.plot(xs, self.train_r2_data, label="Train", color="#2563eb", lw=1.5)
            a2.plot(xs, self.val_r2_data,   label="Val",   color="#ef4444", lw=1.5)
            a2.set_title("R² Score"); a2.legend(); a2.grid(alpha=0.3); a2.set_xlabel("Epoch")
            a2.set_ylim(-0.1, 1.05)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fig.savefig(os.path.join(ckpt_dir, f"train_snapshot_{ts}.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.log_box.append(f"📸 스냅샷 저장: train_snapshot_{ts}.png")
        except Exception as e:
            self.log_box.append(f"[스냅샷 오류] {e}")
