from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QScrollArea, QTextEdit, QProgressBar, QSplitter, QLineEdit, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


# ═══════════════════════════════════════════════════════════════
#  실제 학습 Worker
# ═══════════════════════════════════════════════════════════════
class TrainWorker(QThread):
    epoch_done = pyqtSignal(int, float, float, float, float)
    log_msg    = pyqtSignal(str)
    fold_start = pyqtSignal(int, int)   # fold_idx, total_folds (0,0 = Full 학습)
    finished   = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._stop  = False

    def stop(self):
        self._stop = True

    def _build_model(self, model_name, in_channels, num_classes):
        import segmentation_models_pytorch as smp
        from transformers import (
            SegformerForSemanticSegmentation, SegformerConfig,
            MobileViTForSemanticSegmentation, MobileViTConfig,
            Mask2FormerForUniversalSegmentation, Mask2FormerConfig,
        )

        def _replace_first_conv(model, target_in_channels):
            """3채널 입력층(Conv2d 또는 Linear patch embedding)을 자동 탐색해서 교체"""
            import torch.nn as nn
            if target_in_channels == 3:
                return model

            # ① Conv2d 우선 탐색 (대부분의 CNN/Swin 백본)
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                    parts  = name.split(".")
                    parent = model
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    orig = getattr(parent, parts[-1])
                    setattr(parent, parts[-1], nn.Conv2d(
                        target_in_channels,
                        orig.out_channels,
                        kernel_size=orig.kernel_size,
                        stride=orig.stride,
                        padding=orig.padding,
                        bias=(orig.bias is not None),
                    ))
                    return model

            # ② Linear patch embedding 탐색 (DPT/BEiT/ViT 계열)
            # ViT 계열은 (patch_size × patch_size × 3) → hidden_dim 형태
            # 이름에 "patch_embed" 또는 "embeddings" 포함된 Linear 찾기
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and "patch" in name.lower():
                    in_features = module.in_features
                    # patch_size를 추정 (in_features = patch_size * patch_size * 3)
                    if in_features % 3 == 0:
                        patch_pix      = in_features // 3
                        new_in_features = patch_pix * target_in_channels
                        parts  = name.split(".")
                        parent = model
                        for p in parts[:-1]:
                            parent = getattr(parent, p)
                        orig = getattr(parent, parts[-1])
                        setattr(parent, parts[-1], nn.Linear(
                            new_in_features,
                            orig.out_features,
                            bias=(orig.bias is not None),
                        ))
                        return model

            # ③ 못 찾았으면 경고만 (에러 대신)
            print(f"[Warning] 3채널 입력층을 찾지 못했습니다. "
                  f"이 모델은 {target_in_channels}채널 입력에서 동작 안 할 수 있어요.")
            return model

        SMP_MAP = {
            "UNet": "Unet", "UNet++": "UnetPlusPlus", "MAnet": "MAnet",
            "DeepLabV3": "DeepLabV3", "DeepLabV3+": "DeepLabV3Plus",
            "FPN": "FPN", "PSPNet": "PSPNet", "PAN": "PAN", "Linknet": "Linknet",
        }
        SEGFORMER_CONFIGS = {
            "B0": {"depths": [2,2,2,2], "hidden_sizes": [32,64,160,256]},
            "B2": {"depths": [3,4,6,3], "hidden_sizes": [64,128,320,512]},
            "B4": {"depths": [3,8,27,3], "hidden_sizes": [64,128,320,512]},
        }
        encoder    = self.config.get("encoder", "resnet34")
        pretrained = self.config.get("pretrained", "없음 (scratch)")
        enc_w      = None if "scratch" in pretrained else "imagenet"
        FALLBACK = {
            "기본값 (resnet34)": "resnet34",
            "기본값 (모델 내장)": "resnet34",
            "": "resnet34",
        }
        encoder = FALLBACK.get(encoder, encoder)

        # ── SMP CNN 계열 ─────────────────────────────────────────
        if model_name in SMP_MAP:
            model = getattr(smp, SMP_MAP[model_name])(
                encoder_name=encoder, encoder_weights=enc_w,
                in_channels=in_channels, classes=num_classes,
            )
            return model, False

        # ── SegFormer ────────────────────────────────────────────
        elif "SegFormer" in model_name:
            v = model_name.split("-")[-1]
            vcfg = SEGFORMER_CONFIGS.get(v, SEGFORMER_CONFIGS["B2"])
            cfg = SegformerConfig(num_channels=in_channels, num_labels=num_classes,
                                  depths=vcfg["depths"], hidden_sizes=vcfg["hidden_sizes"])
            return SegformerForSemanticSegmentation(cfg), True

        # ── MobileViT (모바일 친화적 가벼운 모델) ────────────────
        elif model_name == "MobileViT":
            cfg = MobileViTConfig(num_channels=in_channels, num_labels=num_classes)
            model = MobileViTForSemanticSegmentation(cfg)
            model = _replace_first_conv(model, in_channels)
            return model, True

        # ── Mask2Former (Swin 백본) ──────────────────────────────
        elif model_name == "Mask2Former":
            cfg_m2f = Mask2FormerConfig(num_labels=num_classes)
            model   = Mask2FormerForUniversalSegmentation(cfg_m2f)
            model   = _replace_first_conv(model, in_channels)
            return model, True

        # ── fallback ─────────────────────────────────────────────
        else:
            model = smp.Unet(encoder_name="resnet34", encoder_weights=enc_w,
                             in_channels=in_channels, classes=num_classes)
            return model, False

    def _build_optimizer(self, model):
        import torch.optim as optim
        lr   = self.config.get("lr", 6e-4)
        name = self.config.get("optimizer", "AdamW")
        if name == "AdamW":
            return optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.999), weight_decay=0.01)
        elif name == "Adam":
            return optim.Adam(model.parameters(), lr=lr)
        elif name == "SGD":
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        return optim.RMSprop(model.parameters(), lr=lr)

    def _build_scheduler(self, optimizer):
        from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
        name   = self.config.get("scheduler", "ReduceLROnPlateau")
        p1     = self.config.get("sch_param1", 10)
        p2     = self.config.get("sch_param2", 0.5)
        epochs = self.config.get("epochs", 300)
        if name == "StepLR":
            return StepLR(optimizer, step_size=int(p1), gamma=p2), False
        elif name == "CosineAnnealingLR":
            return CosineAnnealingLR(optimizer, T_max=int(p1), eta_min=p2), False
        elif name == "ReduceLROnPlateau":
            return ReduceLROnPlateau(optimizer, mode='min', patience=int(p1), factor=p2), True
        elif name == "OneCycleLR":
            return OneCycleLR(optimizer, max_lr=p2, epochs=epochs, steps_per_epoch=1), False
        return None, False

    def _build_loss(self, device):
        import torch, torch.nn as nn
        import segmentation_models_pytorch as smp
        name  = self.config.get("loss_fn", "CrossEntropyLoss")
        cls_w = self.config.get("class_weights", None)
        w = torch.tensor(cls_w, dtype=torch.float).to(device) if cls_w else None
        if name == "CrossEntropyLoss":
            return nn.CrossEntropyLoss(weight=w)
        elif name == "DiceLoss":
            return smp.losses.DiceLoss(mode='multiclass')
        elif name == "Dice + CE":
            dice = smp.losses.DiceLoss(mode='multiclass')
            ce   = nn.CrossEntropyLoss(weight=w)
            class Combined(nn.Module):
                def forward(self, p, t): return dice(p, t) + ce(p, t)
            return Combined()
        elif name == "FocalLoss":
            return smp.losses.FocalLoss(mode='multiclass')
        elif name == "IoULoss":
            return smp.losses.JaccardLoss(mode='multiclass')
        return nn.CrossEntropyLoss(weight=w)

    def _build_dataset(self):
        import torch, numpy as np, glob, os
        from PIL import Image
        import scipy.io as sio
        from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
        from sklearn.model_selection import StratifiedKFold

        # 모델별 패딩 배수 자동 결정
        try:
            from utils.pad_utils import get_pad_multiple, pad_np
            multiple = get_pad_multiple(self.config.get("model_name","UNet"))
        except Exception:
            multiple = 32
            def pad_np(arr, multiple=32, is_label=False):
                h, w = arr.shape[:2]
                ph = (multiple - h % multiple) % multiple
                pw = (multiple - w % multiple) % multiple
                if ph == 0 and pw == 0:
                    return arr, h, w
                pad_width = ((0,ph),(0,pw),(0,0)) if arr.ndim == 3 else ((0,ph),(0,pw))
                mode = 'constant' if is_label else 'reflect'
                kwargs = {'constant_values': 0} if is_label else {}
                return np.pad(arr, pad_width, mode=mode, **kwargs), h, w

        img_dir = self.config.get("train_img_dir", "")
        lbl_dir = self.config.get("train_lbl_dir", "")
        img_ext = self.config.get("img_ext", "png")
        lbl_ext = self.config.get("lbl_ext", "png")
        batch   = self.config.get("batch", 2)
        split   = self.config.get("split_mode", "manual")

        def load_img(path):
            ext = path.rsplit(".", 1)[-1].lower()
            if ext == "mat":
                return sio.loadmat(path)['image'].astype(np.float32)
            elif ext == "npy":
                return np.load(path).astype(np.float32)
            else:
                return np.array(Image.open(path)).astype(np.float32)

        def load_lbl(path):
            return np.array(Image.open(path))

        # 패딩 적용 + 원본 크기 저장
        orig_size = None
        def pad_imgs_lbls(imgs, lbls):
            nonlocal orig_size
            padded_imgs, padded_lbls = [], []
            for im in imgs:
                p, h, w = pad_np(im, multiple=multiple, is_label=False)
                padded_imgs.append(p)
                if orig_size is None:
                    orig_size = (h, w)
            for lb in lbls:
                p, _, _ = pad_np(lb, multiple=multiple, is_label=True)
                padded_lbls.append(p)
            return padded_imgs, padded_lbls

        def to_tensor(imgs, lbls):
            it = torch.tensor(np.array(imgs)).float()
            lt = torch.tensor(np.array(lbls)).long()
            # (N,H,W,C) → (N,C,H,W)
            if it.ndim == 4 and it.shape[-1] < it.shape[-2]:
                it = it.permute(0, 3, 1, 2)
            elif it.ndim == 3:
                it = it.unsqueeze(1)
            return it, lt

        img_files = sorted(glob.glob(os.path.join(img_dir, f"*.{img_ext}")))
        lbl_files = sorted(glob.glob(os.path.join(lbl_dir, f"*.{lbl_ext}")))
        if not img_files:
            raise FileNotFoundError(f"이미지 없음: {img_dir}/*.{img_ext}")
        if not lbl_files:
            raise FileNotFoundError(f"라벨 없음: {lbl_dir}/*.{lbl_ext}")

        self.log_msg.emit(f"  Train 이미지 {len(img_files)}개 / 라벨 {len(lbl_files)}개")
        imgs = [load_img(p) for p in img_files]
        lbls = [load_lbl(p) for p in lbl_files]

        # 패딩 적용
        imgs, lbls = pad_imgs_lbls(imgs, lbls)
        if orig_size is not None:
            new_h, new_w = imgs[0].shape[:2]
            if (new_h, new_w) != orig_size:
                self.log_msg.emit(
                    f"  자동 패딩: {orig_size[0]}x{orig_size[1]} → {new_h}x{new_w} "
                    f"(multiple={multiple})"
                )

        img_t, lbl_t = to_tensor(imgs, lbls)
        dataset = TensorDataset(img_t, lbl_t)

        # 나중에 config.json에 저장하기 위해 보관
        self.config["_pad_multiple"] = multiple
        self.config["_orig_size"]    = list(orig_size) if orig_size else None

        if split == "manual":
            vi_dir = self.config.get("val_img_dir", "")
            vl_dir = self.config.get("val_lbl_dir", "")
            vi = sorted(glob.glob(os.path.join(vi_dir, f"*.{img_ext}")))
            vl = sorted(glob.glob(os.path.join(vl_dir, f"*.{lbl_ext}")))
            if not vi:
                raise FileNotFoundError(f"Val 이미지 없음: {vi_dir}")
            v_imgs = [load_img(p) for p in vi]
            v_lbls = [load_lbl(p) for p in vl]
            # Val 데이터에도 동일 multiple로 패딩
            v_imgs, v_lbls = pad_imgs_lbls(v_imgs, v_lbls)
            vi_t, vl_t = to_tensor(v_imgs, v_lbls)
            self.log_msg.emit(f"  Val 이미지 {len(vi)}개")
            val_ds = TensorDataset(vi_t, vl_t)
            return [(DataLoader(dataset, batch_size=batch, shuffle=True,
                                num_workers=0, pin_memory=True),
                     DataLoader(val_ds,  batch_size=batch, shuffle=False,
                                num_workers=0, pin_memory=True))]

        elif split == "holdout":
            ratio = self.config.get("holdout_ratio", 0.2)
            n_v   = max(1, int(len(dataset) * ratio))
            n_t   = len(dataset) - n_v
            tr, vl = random_split(dataset, [n_t, n_v])
            self.log_msg.emit(f"  Hold-out: Train {n_t} / Val {n_v}")
            return [(DataLoader(tr, batch_size=batch, shuffle=True,
                                num_workers=0, pin_memory=True),
                     DataLoader(vl, batch_size=batch, shuffle=False,
                                num_workers=0, pin_memory=True))]

        elif split == "kfold":
            k   = self.config.get("kfold_k", 5)
            dom = [int(lbl_t[i].flatten().mode()[0]) for i in range(len(lbl_t))]
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            folds = []
            X = list(range(len(dataset)))
            for tr_idx, vl_idx in skf.split(X, dom):
                folds.append((
                    DataLoader(Subset(dataset, tr_idx), batch_size=batch, shuffle=True,
                               num_workers=0, pin_memory=True),
                    DataLoader(Subset(dataset, vl_idx), batch_size=batch, shuffle=False,
                               num_workers=0, pin_memory=True),
                ))
            self.log_msg.emit(f"  Stratified K-Fold: K={k}")
            return folds
        return []

    def _augment(self, data, labels):
        import torch
        aug = self.config.get("augmentation", {})
        if aug.get("RandomHorizontalFlip") and torch.rand(1) < aug.get("RandomHorizontalFlip_val", 0.5):
            data = torch.flip(data, [-1]); labels = torch.flip(labels, [-1])
        if aug.get("RandomVerticalFlip") and torch.rand(1) < aug.get("RandomVerticalFlip_val", 0.5):
            data = torch.flip(data, [-2]); labels = torch.flip(labels, [-2])
        return data, labels

    @staticmethod
    def _get_logits(out, is_tr, model_name, labels):
        """모델 출력에서 (N,C,H,W) logits 추출 — Mask2Former 특수 처리 포함"""
        import torch, torch.nn.functional as F

        if "Mask2Former" in (model_name or ""):
            # Mask2Former: class_queries_logits (N,Q,C) × masks_queries_logits (N,Q,H,W)
            cls_logits  = out.class_queries_logits   # (N, Q, num_labels+1)
            mask_logits = out.masks_queries_logits   # (N, Q, H', W')
            # 각 픽셀에 대해 query별 class score × mask score 합산
            cls_prob  = cls_logits[:, :, :-1].softmax(-1)   # (N, Q, C)  마지막은 no-obj
            mask_prob = mask_logits.sigmoid()                # (N, Q, H', W')
            # einsum: N Q C, N Q H W -> N C H W
            seg = torch.einsum("bqc,bqhw->bchw", cls_prob, mask_prob)
            if seg.shape[-2:] != labels.shape[-2:]:
                seg = F.interpolate(seg, size=labels.shape[1:],
                                    mode='bilinear', align_corners=False)
            return seg

        logits = out.logits if is_tr else out
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[1:],
                                   mode='bilinear', align_corners=False)
        return logits

    @staticmethod
    def _calc_acc(logits, labels):
        import torch
        return (torch.argmax(logits, dim=1) == labels).float().mean().item()

    @staticmethod
    def _resize(logits, labels):
        import torch.nn.functional as F
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[1:],
                                   mode='bilinear', align_corners=False)
        return logits

    def run(self):
        try:
            self._run_train()
        except Exception as e:
            import traceback
            self.log_msg.emit(f"[오류] {e}\n{traceback.format_exc()}")
        finally:
            self.finished.emit()

    def _run_train(self):
        import os, json, torch, numpy as np
        cfg      = self.config
        epochs   = cfg.get("epochs", 300)
        ckpt_dir = cfg.get("ckpt_dir", "./checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_msg.emit(f"[학습 시작]  디바이스: {device}")
        self.log_msg.emit(f"  모델: {cfg.get('model_name','?')}  |  Epochs: {epochs}  |  LR: {cfg.get('lr')}")
        self.log_msg.emit(f"  저장 경로: {ckpt_dir}")

        folds = self._build_dataset()
        if not folds:
            self.log_msg.emit("[오류] 데이터셋 생성 실패 — 데이터 탭 경로를 확인하세요")
            return

        in_ch      = cfg.get("in_channels", 3)
        n_cls      = cfg.get("num_classes", 2)
        is_kfold   = cfg.get("split_mode") == "kfold" and len(folds) > 1
        fold_results = []   # K-Fold 각 fold 결과 수집

        # ── K-Fold / Hold-out / Manual 학습 루프 ─────────────────
        for fold_idx, (train_loader, val_loader) in enumerate(folds):
            if self._stop: break

            fold_tag = f"[Fold {fold_idx+1}/{len(folds)}] " if len(folds) > 1 else ""
            if fold_tag:
                self.log_msg.emit(f"\n── Fold {fold_idx+1}/{len(folds)} 시작 ──")

            # Fold 시작 시 플롯 초기화 시그널
            self.fold_start.emit(fold_idx + 1, len(folds))

            model, is_tr = self._build_model(cfg.get("model_name","UNet"), in_ch, n_cls)
            model        = model.to(device)
            optimizer    = self._build_optimizer(model)
            scheduler, sch_on_loss = self._build_scheduler(optimizer)
            criterion    = self._build_loss(device)

            # AMP (Automatic Mixed Precision) — GPU에서 1.5~3배 속도 향상
            use_amp = (device.type == "cuda")
            scaler  = torch.amp.GradScaler('cuda') if use_amp else None
            if use_amp and fold_idx == 0:
                self.log_msg.emit(f"  AMP (Mixed Precision) 활성화 — GPU 학습 가속")

            use_es      = cfg.get("use_early_stopping", True)
            es_patience = cfg.get("es_patience", 50)
            es_counter  = 0
            best_vl     = float('inf')
            best_va     = 0.0

            for ep in range(1, epochs + 1):
                if self._stop:
                    self.log_msg.emit("[중단]"); break

                model.train()
                tl = ta = 0.0
                model_name = cfg.get("model_name","UNet")
                for bd, bl in train_loader:
                    if torch.rand(1) < 0.5:
                        bd, bl = self._augment(bd, bl)
                    bd = bd.to(device, non_blocking=True)
                    bl = bl.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            out    = model(bd)
                            logits = self._get_logits(out, is_tr, model_name, bl)
                            loss   = criterion(logits, bl)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        out    = model(bd)
                        logits = self._get_logits(out, is_tr, model_name, bl)
                        loss   = criterion(logits, bl)
                        loss.backward()
                        optimizer.step()

                    acc = self._calc_acc(logits, bl)
                    tl += loss.item(); ta += acc
                avg_tl = tl / len(train_loader)
                avg_ta = ta / len(train_loader)

                model.eval()
                vl = va = 0.0
                with torch.no_grad():
                    for bd, bl in val_loader:
                        bd = bd.to(device, non_blocking=True)
                        bl = bl.to(device, non_blocking=True)
                        if use_amp:
                            with torch.amp.autocast('cuda'):
                                out    = model(bd)
                                logits = self._get_logits(out, is_tr, model_name, bl)
                                loss   = criterion(logits, bl)
                        else:
                            out    = model(bd)
                            logits = self._get_logits(out, is_tr, model_name, bl)
                            loss   = criterion(logits, bl)
                        vl += loss.item()
                        va += self._calc_acc(logits, bl)
                avg_vl = vl / len(val_loader)
                avg_va = va / len(val_loader)

                if scheduler:
                    scheduler.step(avg_vl) if sch_on_loss else scheduler.step()
                cur_lr = optimizer.param_groups[0]['lr']

                if avg_vl < best_vl:
                    best_vl    = avg_vl
                    best_va    = avg_va
                    es_counter = 0
                    suffix     = f"_fold{fold_idx+1}" if len(folds) > 1 else ""
                    best_path  = os.path.join(ckpt_dir, f"best_model{suffix}.pth")
                    torch.save(model.state_dict(), best_path)

                    # 모델 구조 정보 json — 추후 로드 시 자동 복원용
                    model_config = {
                        "model_name":   cfg.get("model_name","UNet"),
                        "encoder":      cfg.get("encoder","resnet34"),
                        "in_channels":  cfg.get("in_channels",3),
                        "num_classes":  cfg.get("num_classes",2),
                        "pad_multiple": cfg.get("_pad_multiple", 32),
                        "orig_size":    cfg.get("_orig_size", None),
                        "fold":         fold_idx + 1 if len(folds) > 1 else None,
                        "best_epoch":   ep,
                        "val_loss":     round(avg_vl,6),
                        "val_acc":      round(avg_va,6),
                    }
                    config_path = best_path.replace(".pth", "_config.json")
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(model_config, f, indent=2, ensure_ascii=False)

                    self.log_msg.emit(
                        f"  ✔ Best ep{ep} | val_loss={avg_vl:.4f} val_acc={avg_va*100:.2f}%"
                    )
                else:
                    es_counter += 1

                self.epoch_done.emit(ep, avg_tl, avg_vl, avg_ta, avg_va)
                self.log_msg.emit(
                    f"{fold_tag}Ep {ep:03d}/{epochs} | "
                    f"Loss {avg_tl:.4f}/{avg_vl:.4f} | "
                    f"Acc {avg_ta*100:.1f}%/{avg_va*100:.1f}% | "
                    f"LR {cur_lr:.2e} | ES {es_counter}/{es_patience}"
                )

                if use_es and es_counter >= es_patience:
                    self.log_msg.emit(f"  Early Stopping! (ep{ep})")
                    break

            # Fold 결과 기록
            fold_results.append({
                "fold":      fold_idx + 1,
                "best_val_loss": round(best_vl, 6),
                "best_val_acc":  round(best_va, 6),
                "best_model":    best_path,
            })

        # ── K-Fold 결과 요약 저장 ─────────────────────────────────
        if is_kfold and not self._stop:
            accs   = [r["best_val_acc"]  for r in fold_results]
            losses = [r["best_val_loss"] for r in fold_results]
            summary = {
                "model":        cfg.get("model_name","?"),
                "encoder":      cfg.get("encoder","?"),
                "k_folds":      len(folds),
                "epochs":       epochs,
                "fold_results": fold_results,
                "mean_val_acc":  round(float(np.mean(accs)),  6),
                "std_val_acc":   round(float(np.std(accs)),   6),
                "mean_val_loss": round(float(np.mean(losses)),6),
                "std_val_loss":  round(float(np.std(losses)), 6),
            }
            summary_path = os.path.join(ckpt_dir, "kfold_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            self.log_msg.emit(f"\n{'='*50}")
            self.log_msg.emit(f"[K-Fold 결과 요약]")
            for r in fold_results:
                self.log_msg.emit(
                    f"  Fold {r['fold']}: val_acc={r['best_val_acc']*100:.2f}%  "
                    f"val_loss={r['best_val_loss']:.4f}"
                )
            self.log_msg.emit(f"  평균 Acc : {summary['mean_val_acc']*100:.2f}% "
                              f"± {summary['std_val_acc']*100:.2f}%")
            self.log_msg.emit(f"  평균 Loss: {summary['mean_val_loss']:.4f} "
                              f"± {summary['std_val_loss']:.4f}")
            self.log_msg.emit(f"  저장: {summary_path}")

            # ── Full Data 학습 ────────────────────────────────────
            if not self._stop:
                self.log_msg.emit(f"\n{'='*50}")
                self.log_msg.emit(f"[Full Data 학습 시작]  전체 데이터로 최종 모델 생성")
                self.fold_start.emit(0, 0)  # 플롯 초기화

                full_loader, _ = self._build_full_dataset()
                if full_loader:
                    model_f, is_tr_f = self._build_model(
                        cfg.get("model_name","UNet"), in_ch, n_cls)
                    model_f    = model_f.to(device)
                    optimizer_f = self._build_optimizer(model_f)
                    scheduler_f, sch_on_loss_f = self._build_scheduler(optimizer_f)
                    criterion_f = self._build_loss(device)
                    best_full_loss = float('inf')
                    use_amp_f = (device.type == "cuda")
                    scaler_f  = torch.amp.GradScaler('cuda') if use_amp_f else None

                    for ep in range(1, epochs + 1):
                        if self._stop: break
                        model_f.train()
                        tl = ta = 0.0
                        for bd, bl in full_loader:
                            if torch.rand(1) < 0.5:
                                bd, bl = self._augment(bd, bl)
                            bd = bd.to(device, non_blocking=True)
                            bl = bl.to(device, non_blocking=True)

                            optimizer_f.zero_grad(set_to_none=True)
                            if use_amp_f:
                                with torch.amp.autocast('cuda'):
                                    out    = model_f(bd)
                                    logits = self._get_logits(out, is_tr_f,
                                                              cfg.get("model_name","UNet"), bl)
                                    loss   = criterion_f(logits, bl)
                                scaler_f.scale(loss).backward()
                                scaler_f.step(optimizer_f)
                                scaler_f.update()
                            else:
                                out    = model_f(bd)
                                logits = self._get_logits(out, is_tr_f,
                                                          cfg.get("model_name","UNet"), bl)
                                loss   = criterion_f(logits, bl)
                                loss.backward()
                                optimizer_f.step()

                            acc = self._calc_acc(logits, bl)
                            tl += loss.item(); ta += acc
                        avg_tl = tl / len(full_loader)
                        avg_ta = ta / len(full_loader)

                        if scheduler_f:
                            scheduler_f.step(avg_tl) if sch_on_loss_f else scheduler_f.step()
                        cur_lr = optimizer_f.param_groups[0]['lr']

                        if avg_tl < best_full_loss:
                            best_full_loss = avg_tl
                            full_path = os.path.join(ckpt_dir, "best_model_full.pth")
                            torch.save(model_f.state_dict(), full_path)
                            # Full 모델 config.json
                            full_config = {
                                "model_name":   cfg.get("model_name","UNet"),
                                "encoder":      cfg.get("encoder","resnet34"),
                                "in_channels":  cfg.get("in_channels",3),
                                "num_classes":  cfg.get("num_classes",2),
                                "pad_multiple": cfg.get("_pad_multiple", 32),
                                "orig_size":    cfg.get("_orig_size", None),
                                "fold":         "full",
                                "best_epoch":   ep,
                                "train_loss":   round(avg_tl,6),
                                "train_acc":    round(avg_ta,6),
                            }
                            with open(full_path.replace(".pth","_config.json"),
                                      "w", encoding="utf-8") as f:
                                json.dump(full_config, f, indent=2, ensure_ascii=False)

                        self.epoch_done.emit(ep, avg_tl, avg_tl, avg_ta, avg_ta)
                        self.log_msg.emit(
                            f"[Full] Ep {ep:03d}/{epochs} | "
                            f"Loss {avg_tl:.4f} | Acc {avg_ta*100:.1f}% | LR {cur_lr:.2e}"
                        )

                    self.log_msg.emit(f"  ✔ Full 모델 저장: {full_path}")
                    # summary에 full 결과 추가
                    summary["full_model_path"] = full_path
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)

        # ── 최종 ─────────────────────────────────────────────────
        self.log_msg.emit(f"\n✔ 전체 학습 완료!")

    def _build_full_dataset(self):
        """K-Fold용 전체 데이터 DataLoader 생성"""
        import torch, numpy as np, glob, os
        from PIL import Image
        import scipy.io as sio
        from torch.utils.data import DataLoader, TensorDataset

        try:
            from utils.pad_utils import get_pad_multiple, pad_np
            multiple = get_pad_multiple(self.config.get("model_name","UNet"))
        except Exception:
            multiple = 32
            def pad_np(arr, multiple=32, is_label=False):
                h, w = arr.shape[:2]
                ph = (multiple - h % multiple) % multiple
                pw = (multiple - w % multiple) % multiple
                if ph == 0 and pw == 0:
                    return arr, h, w
                pad_width = ((0,ph),(0,pw),(0,0)) if arr.ndim == 3 else ((0,ph),(0,pw))
                mode = 'constant' if is_label else 'reflect'
                kwargs = {'constant_values': 0} if is_label else {}
                return np.pad(arr, pad_width, mode=mode, **kwargs), h, w

        cfg     = self.config
        img_dir = cfg.get("train_img_dir","")
        lbl_dir = cfg.get("train_lbl_dir","")
        img_ext = cfg.get("img_ext","png")
        lbl_ext = cfg.get("lbl_ext","png")
        batch   = cfg.get("batch",2)

        def load_img(p):
            ext = p.rsplit(".",1)[-1].lower()
            if ext=="mat":   return sio.loadmat(p)['image'].astype(np.float32)
            elif ext=="npy": return np.load(p).astype(np.float32)
            else:            return np.array(Image.open(p)).astype(np.float32)

        img_files = sorted(glob.glob(os.path.join(img_dir, f"*.{img_ext}")))
        lbl_files = sorted(glob.glob(os.path.join(lbl_dir, f"*.{lbl_ext}")))
        if not img_files: return None, None

        imgs_raw = [load_img(p) for p in img_files]
        lbls_raw = [np.array(Image.open(p)) for p in lbl_files]
        # 패딩 적용
        imgs = [pad_np(im, multiple=multiple, is_label=False)[0] for im in imgs_raw]
        lbls = [pad_np(lb, multiple=multiple, is_label=True)[0]  for lb in lbls_raw]

        it = torch.tensor(np.array(imgs)).float()
        lt = torch.tensor(np.array(lbls)).long()
        if it.ndim==4 and it.shape[-1]<it.shape[-2]: it = it.permute(0,3,1,2)
        elif it.ndim==3: it = it.unsqueeze(1)
        ds = TensorDataset(it, lt)
        self.log_msg.emit(f"  Full 데이터: {len(ds)}장 (multiple={multiple})")
        return DataLoader(ds, batch_size=batch, shuffle=True,
                          num_workers=0, pin_memory=True), None


# ═══════════════════════════════════════════════════════════════
#  TrainTab UI
# ═══════════════════════════════════════════════════════════════
class TrainTab(QWidget):
    def __init__(self, shared_state):
        super().__init__()
        self.state  = shared_state
        self.worker = None
        self._data_tab = None
        self.train_loss_data = []
        self.val_loss_data   = []
        self.train_acc_data  = []
        self.val_acc_data    = []
        self._build()

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)

        left_w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner  = QWidget(); scroll.setWidget(inner)
        left_v = QVBoxLayout(inner)
        left_v.setSpacing(12); left_v.setContentsMargins(16, 16, 16, 16)

        # 하이퍼파라미터
        hp = QGroupBox("하이퍼파라미터")
        hg = QGridLayout(hp)
        hg.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs = QSpinBox(); self.epochs.setRange(1,5000); self.epochs.setValue(300)
        hg.addWidget(self.epochs, 0, 1)
        hg.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch = QSpinBox(); self.batch.setRange(1,256); self.batch.setValue(2)
        hg.addWidget(self.batch, 1, 1)
        hg.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(6); self.lr.setRange(1e-7,1.0)
        self.lr.setValue(0.0006); self.lr.setSingleStep(0.0001)
        hg.addWidget(self.lr, 2, 1)
        hg.addWidget(QLabel("Optimizer:"), 3, 0)
        self.optimizer = QComboBox()
        self.optimizer.addItems(["AdamW","Adam","SGD","RMSprop"])
        hg.addWidget(self.optimizer, 3, 1)
        hg.addWidget(QLabel("Loss Function:"), 4, 0)
        self.loss_fn = QComboBox()
        self.loss_fn.addItems(["CrossEntropyLoss","DiceLoss","Dice + CE","FocalLoss","IoULoss"])
        hg.addWidget(self.loss_fn, 4, 1)
        left_v.addWidget(hp)

        # LR Scheduler
        sch_g = QGroupBox("LR Scheduler")
        sg = QGridLayout(sch_g)
        sg.addWidget(QLabel("Scheduler:"), 0, 0)
        self.scheduler = QComboBox()
        self.scheduler.addItems(["없음","StepLR","CosineAnnealingLR","ReduceLROnPlateau","OneCycleLR"])
        self.scheduler.setCurrentText("ReduceLROnPlateau")
        self.scheduler.currentTextChanged.connect(self._on_scheduler_changed)
        sg.addWidget(self.scheduler, 0, 1)
        self.sch_p1_lbl = QLabel("Patience:"); sg.addWidget(self.sch_p1_lbl, 1, 0)
        self.sch_p1 = QSpinBox(); self.sch_p1.setRange(1,1000); self.sch_p1.setValue(10)
        sg.addWidget(self.sch_p1, 1, 1)
        self.sch_p2_lbl = QLabel("Factor:"); sg.addWidget(self.sch_p2_lbl, 2, 0)
        self.sch_p2 = QDoubleSpinBox(); self.sch_p2.setDecimals(3)
        self.sch_p2.setRange(0.001,1.0); self.sch_p2.setValue(0.5)
        sg.addWidget(self.sch_p2, 2, 1)
        left_v.addWidget(sch_g)

        # Early Stopping
        es_g = QGroupBox("Early Stopping")
        eg = QGridLayout(es_g)
        self.use_es = QCheckBox("Early Stopping 사용"); self.use_es.setChecked(True)
        eg.addWidget(self.use_es, 0, 0, 1, 2)
        eg.addWidget(QLabel("Patience:"), 1, 0)
        self.es_patience = QSpinBox(); self.es_patience.setRange(1,500); self.es_patience.setValue(50)
        eg.addWidget(self.es_patience, 1, 1)
        left_v.addWidget(es_g)

        # Augmentation
        aug_g = QGroupBox("Augmentation")
        ag = QGridLayout(aug_g)
        ag.addWidget(QLabel("항목"), 0, 0)
        ag.addWidget(QLabel("사용"), 0, 1)
        ag.addWidget(QLabel("수치"), 0, 2)
        aug_defs = [
            ("RandomHorizontalFlip", True,  0.5,  "p",       "float"),
            ("RandomVerticalFlip",   True,  0.5,  "p",       "float"),
            ("RandomRotation",       True,  30.0, "degree",  "float"),
            ("ColorJitter",          True,  0.2,  "strength","float"),
            ("RandomCrop",           False, 0.85, "scale",   "float"),
            ("GaussianBlur",         False, 1.0,  "σ",       "float"),
            ("RandomGrayscale",      False, 0.1,  "p",       "float"),
            ("Normalize",            True,  0.0,  "(자동)",   "none"),
        ]
        self.aug_checks = {}; self.aug_spins = {}
        for i, (name, on, val, lbl, typ) in enumerate(aug_defs):
            r = i + 1
            ag.addWidget(QLabel(name), r, 0)
            cb = QCheckBox(); cb.setChecked(on)
            ag.addWidget(cb, r, 1); self.aug_checks[name] = cb
            if typ == "float":
                sp = QDoubleSpinBox(); sp.setDecimals(2); sp.setRange(0.0,360.0)
                sp.setValue(val); sp.setPrefix(f"{lbl}="); sp.setFixedWidth(110)
                ag.addWidget(sp, r, 2); self.aug_spins[name] = sp
            else:
                ag.addWidget(QLabel("—"), r, 2)
        left_v.addWidget(aug_g)

        # Checkpoint
        ck_g = QGroupBox("Checkpoint 저장 경로")
        ck = QGridLayout(ck_g)
        self.ckpt_path_edit = QLineEdit(); self.ckpt_path_edit.setPlaceholderText("./checkpoints")
        ck.addWidget(self.ckpt_path_edit, 0, 0)
        btn_ck = QPushButton("📁"); btn_ck.setObjectName("secondary"); btn_ck.setFixedWidth(36)
        btn_ck.clicked.connect(self._browse_ckpt)
        ck.addWidget(btn_ck, 0, 1)
        left_v.addWidget(ck_g)
        left_v.addStretch()

        lw = QVBoxLayout(left_w); lw.setContentsMargins(0,0,0,0); lw.addWidget(scroll)
        splitter.addWidget(left_w)

        # 오른쪽 모니터
        right_w = QWidget()
        rv = QVBoxLayout(right_w); rv.setContentsMargins(16,16,16,16); rv.setSpacing(8)

        btn_h = QHBoxLayout()
        self.start_btn = QPushButton("▶  학습 시작"); self.start_btn.clicked.connect(self._start_train)
        self.stop_btn  = QPushButton("■  중단"); self.stop_btn.setObjectName("secondary")
        self.stop_btn.clicked.connect(self._stop_train); self.stop_btn.setEnabled(False)
        btn_h.addWidget(self.start_btn); btn_h.addWidget(self.stop_btn)
        rv.addLayout(btn_h)

        self.progress = QProgressBar(); self.progress.setRange(0,100); rv.addWidget(self.progress)
        self.epoch_label = QLabel("대기 중..."); self.epoch_label.setObjectName("hint")
        rv.addWidget(self.epoch_label)

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
            from matplotlib.figure import Figure
            self.fig = Figure(figsize=(6,4), facecolor='white')
            self.ax1 = self.fig.add_subplot(121); self.ax2 = self.fig.add_subplot(122)
            self.canvas = Canvas(self.fig); rv.addWidget(self.canvas); self.has_mpl = True
        except ImportError:
            rv.addWidget(QLabel("matplotlib 미설치")); self.has_mpl = False

        log_lbl = QLabel("학습 로그"); log_lbl.setObjectName("section_title"); rv.addWidget(log_lbl)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(180)
        rv.addWidget(self.log_box)

        splitter.addWidget(right_w); splitter.setSizes([380,620]); outer.addWidget(splitter)

    def _on_scheduler_changed(self, name):
        param_map = {
            "StepLR":            ("Step Size:", "Gamma:"),
            "CosineAnnealingLR": ("T_max:",     "eta_min:"),
            "ReduceLROnPlateau": ("Patience:",  "Factor:"),
            "OneCycleLR":        ("Max LR:",    "Pct Start:"),
            "없음":              ("",           ""),
        }
        p1, p2 = param_map.get(name, ("Param1:", "Param2:"))
        self.sch_p1_lbl.setText(p1); self.sch_p2_lbl.setText(p2)
        visible = name != "없음"
        for w in [self.sch_p1_lbl, self.sch_p1, self.sch_p2_lbl, self.sch_p2]:
            w.setVisible(visible)

    def _browse_ckpt(self):
        path = QFileDialog.getExistingDirectory(self, "저장 폴더")
        if path: self.ckpt_path_edit.setText(path)

    def set_data_tab(self, data_tab):
        """main.py에서 호출 — 데이터탭 직접 참조로 저장 버튼 없이도 경로 읽기"""
        self._data_tab = data_tab

    def _get_data_config(self):
        """shared_state 우선, 없으면 data_tab UI에서 직접 읽기"""
        # shared_state에 경로가 있으면 그걸 씀
        if self.state.get("train_img_dir", "").strip():
            return {
                "train_img_dir": self.state.get("train_img_dir", ""),
                "train_lbl_dir": self.state.get("train_lbl_dir", ""),
                "val_img_dir":   self.state.get("val_img_dir", ""),
                "val_lbl_dir":   self.state.get("val_lbl_dir", ""),
                "img_ext":       self.state.get("img_ext", "png"),
                "lbl_ext":       self.state.get("lbl_ext", "png"),
                "split_mode":    self.state.get("split_mode", "manual"),
                "holdout_ratio": self.state.get("holdout_ratio", 0.2),
                "kfold_k":       self.state.get("kfold_k", 5),
            }
        # data_tab UI에서 직접 읽기 (저장 버튼 안 눌렀어도 동작)
        if hasattr(self, '_data_tab') and self._data_tab is not None:
            dt = self._data_tab
            mode_map = {0: "manual", 1: "holdout", 2: "kfold"}
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
            }
        return {}

    def _start_train(self):
        self.train_loss_data.clear(); self.val_loss_data.clear()
        self.train_acc_data.clear();  self.val_acc_data.clear()
        self.log_box.clear(); self.progress.setValue(0)

        # 데이터 경로 읽기 (저장 버튼 없이도 동작)
        data_cfg = self._get_data_config()

        # 경로 검증
        train_img = data_cfg.get("train_img_dir", "").strip()
        train_lbl = data_cfg.get("train_lbl_dir", "").strip()
        split_mode = data_cfg.get("split_mode", "manual")

        if not train_img:
            self.log_box.append("[오류] Train 이미지 폴더가 비어있어요\n"
                                "→ ① 데이터 탭에서 경로를 설정하거나\n"
                                "→ '데이터 설정 저장' 버튼을 눌러주세요")
            return
        if not train_lbl:
            self.log_box.append("[오류] Train 라벨 폴더가 비어있어요")
            return
        if split_mode == "manual":
            val_img = data_cfg.get("val_img_dir", "").strip()
            if not val_img:
                self.log_box.append("[오류] Val 이미지 폴더가 비어있어요\n"
                                    "→ Hold-out 또는 K-Fold 방식으로 변경하거나\n"
                                    "→ Val 폴더를 지정해주세요")
                return

        aug_cfg = {}
        for name, cb in self.aug_checks.items():
            aug_cfg[name] = cb.isChecked()
            if name in self.aug_spins:
                aug_cfg[f"{name}_val"] = self.aug_spins[name].value()

        config = {
            **data_cfg,
            "model_name":         self.state.get("model_name", "UNet"),
            "encoder":            self.state.get("encoder", "resnet34"),
            "pretrained":         self.state.get("pretrained", "없음 (scratch)"),
            "in_channels":        self.state.get("in_channels", 3),
            "num_classes":        self.state.get("num_classes", 2),
            "class_weights":      self.state.get("class_weights", None),
            "epochs":             self.epochs.value(),
            "batch":              self.batch.value(),
            "lr":                 self.lr.value(),
            "optimizer":          self.optimizer.currentText(),
            "loss_fn":            self.loss_fn.currentText(),
            "scheduler":          self.scheduler.currentText(),
            "sch_param1":         self.sch_p1.value(),
            "sch_param2":         self.sch_p2.value(),
            "use_early_stopping": self.use_es.isChecked(),
            "es_patience":        self.es_patience.value(),
            "augmentation":       aug_cfg,
            "ckpt_dir":           self.ckpt_path_edit.text().strip() or "./checkpoints",
        }

        self.log_box.append(f"[경로 확인]")
        self.log_box.append(f"  Train 이미지: {config['train_img_dir']}")
        self.log_box.append(f"  Train 라벨:   {config['train_lbl_dir']}")
        self.log_box.append(f"  분할 방식:    {config['split_mode']}")
        self.total_epochs = config["epochs"]
        self.worker = TrainWorker(config)
        self.worker.epoch_done.connect(self._on_epoch)
        self.worker.fold_start.connect(self._on_fold_start)
        self.worker.log_msg.connect(self.log_box.append)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)

    def _on_fold_start(self, fold_idx, total):
        """Fold 시작 시 플롯 초기화"""
        self.train_loss_data.clear(); self.val_loss_data.clear()
        self.train_acc_data.clear();  self.val_acc_data.clear()
        self.progress.setValue(0)
        if total == 0:
            self.epoch_label.setText("Full Data 학습 중...")
        else:
            self.epoch_label.setText(f"Fold {fold_idx} / {total} 학습 중...")
        if self.has_mpl:
            self.ax1.cla(); self.ax2.cla()
            self.canvas.draw()

    def _stop_train(self):
        if self.worker: self.worker.stop()

    def _on_epoch(self, ep, tl, vl, ta, va):
        self.train_loss_data.append(tl); self.val_loss_data.append(vl)
        self.train_acc_data.append(ta);  self.val_acc_data.append(va)
        self.progress.setValue(int(ep / self.total_epochs * 100))
        self.epoch_label.setText(
            f"Epoch {ep}/{self.total_epochs}  |  Val Loss: {vl:.4f}  Val Acc: {va*100:.2f}%"
        )
        if self.has_mpl: self._update_plot()

    def _update_plot(self):
        self.ax1.cla(); self.ax2.cla()
        xs = list(range(1, len(self.train_loss_data)+1))
        self.ax1.plot(xs, self.train_loss_data, label="Train", color="#2563eb", lw=1.5)
        self.ax1.plot(xs, self.val_loss_data,   label="Val",   color="#ef4444", lw=1.5)
        self.ax1.set_title("Loss", fontsize=10); self.ax1.legend(fontsize=8); self.ax1.grid(alpha=0.3)
        self.ax2.plot(xs, self.train_acc_data, label="Train", color="#2563eb", lw=1.5)
        self.ax2.plot(xs, self.val_acc_data,   label="Val",   color="#ef4444", lw=1.5)
        self.ax2.set_title("Accuracy", fontsize=10); self.ax2.legend(fontsize=8)
        self.ax2.set_ylim(0,1); self.ax2.grid(alpha=0.3)
        self.fig.tight_layout(); self.canvas.draw()

    def _on_finished(self):
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.log_box.append("✔ 학습 완료"); self.progress.setValue(100)
        self._save_train_snapshot()

    def _save_train_snapshot(self):
        """학습 Loss/Accuracy 플롯 + 하이퍼파라미터를 checkpoint 폴더에 저장"""
        import os, json, datetime
        ckpt_dir = self.ckpt_path_edit.text().strip() or "./checkpoints"
        if not self.has_mpl or not self.train_loss_data:
            return
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            for font in ["Malgun Gothic","AppleGothic","NanumGothic","DejaVu Sans"]:
                try:
                    matplotlib.rcParams['font.family'] = font
                    matplotlib.rcParams['axes.unicode_minus'] = False
                    break
                except Exception: continue

            xs = list(range(1, len(self.train_loss_data)+1))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.plot(xs, self.train_loss_data, label="Train Loss", color="#2563eb", lw=1.5)
            ax1.plot(xs, self.val_loss_data,   label="Val Loss",   color="#ef4444", lw=1.5)
            ax1.set_title("Loss", fontsize=12); ax1.legend(); ax1.grid(alpha=0.3)
            ax1.set_xlabel("Epoch")

            ax2.plot(xs, self.train_acc_data, label="Train Acc", color="#2563eb", lw=1.5)
            ax2.plot(xs, self.val_acc_data,   label="Val Acc",   color="#ef4444", lw=1.5)
            ax2.set_title("Accuracy", fontsize=12); ax2.legend()
            ax2.set_ylim(0,1); ax2.grid(alpha=0.3); ax2.set_xlabel("Epoch")

            # 하이퍼파라미터 텍스트 박스
            params_text = "\n".join([
                f"Model: {self.state.get('model_name','?')}",
                f"Encoder: {self.state.get('encoder','?')}",
                f"In_channels: {self.state.get('in_channels','?')}",
                f"Classes: {self.state.get('num_classes','?')}",
                f"Epochs: {self.epochs.value()}",
                f"Batch: {self.batch.value()}",
                f"LR: {self.lr.value():.6f}",
                f"Optimizer: {self.optimizer.currentText()}",
                f"Loss: {self.loss_fn.currentText()}",
                f"Scheduler: {self.scheduler.currentText()}",
                f"ES Patience: {self.es_patience.value()}",
                f"Split: {self.state.get('split_mode','?')}",
            ])
            fig.text(0.5, -0.05, params_text, ha='center', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.8))

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(ckpt_dir, f"train_snapshot_{timestamp}.png")
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.log_box.append(f"📸 학습 스냅샷 저장: {plot_path}")

            # 하이퍼파라미터 json 별도 저장
            meta = {
                "timestamp": timestamp,
                "model_name": self.state.get("model_name","?"),
                "encoder":    self.state.get("encoder","?"),
                "in_channels": self.state.get("in_channels","?"),
                "num_classes": self.state.get("num_classes","?"),
                "epochs":     self.epochs.value(),
                "batch_size": self.batch.value(),
                "lr":         self.lr.value(),
                "optimizer":  self.optimizer.currentText(),
                "loss_fn":    self.loss_fn.currentText(),
                "scheduler":  self.scheduler.currentText(),
                "sch_param1": self.sch_p1.value(),
                "sch_param2": self.sch_p2.value(),
                "early_stopping": self.use_es.isChecked(),
                "es_patience":    self.es_patience.value(),
                "split_mode": self.state.get("split_mode","?"),
                "final_train_loss": round(self.train_loss_data[-1],6) if self.train_loss_data else None,
                "final_val_loss":   round(self.val_loss_data[-1],6)   if self.val_loss_data   else None,
                "final_train_acc":  round(self.train_acc_data[-1],6)  if self.train_acc_data  else None,
                "final_val_acc":    round(self.val_acc_data[-1],6)    if self.val_acc_data    else None,
                "augmentation": {k: cb.isChecked() for k,cb in self.aug_checks.items()},
            }
            json_path = os.path.join(ckpt_dir, f"hyperparams_{timestamp}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            self.log_box.append(f"📋 하이퍼파라미터 저장: {json_path}")

        except Exception as e:
            self.log_box.append(f"[스냅샷 오류] {e}")
