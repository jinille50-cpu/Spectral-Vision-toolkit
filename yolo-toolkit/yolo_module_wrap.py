"""
YOLO Detection 학습/데이터 래퍼

기존 YOLO_Module.py(YOLOproject)를 기반으로,
GUI에서 쓰기 좋게 다음을 추가:
  - scan_dataset()  : 이미지/라벨 검증 (개수·짝·클래스 ID)
  - build_yaml()    : data.yaml 생성 (manual 폴더 분리 / hold-out 랜덤 분할)
  - train()         : Ultralytics 콜백으로 epoch별 metric 추출 + 중단 지원

라벨은 YOLO Detection txt 형식(class cx cy w h)만 사용.
이미지는 png / tiff / tif.
"""

import os
import glob
import shutil
import random


IMG_EXTS = ("png", "tiff", "tif")


# ──────────────────────────────────────────────────────────────────
#  데이터 검증
# ──────────────────────────────────────────────────────────────────
def _list_images(img_dir, img_ext):
    return sorted(glob.glob(os.path.join(img_dir, f"*.{img_ext}")))


def _label_for(img_path, lbl_dir):
    """이미지 경로 → 같은 stem의 .txt 라벨 경로"""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(lbl_dir, stem + ".txt")


def scan_dataset(img_dir, lbl_dir, img_ext="png"):
    """
    이미지/라벨 폴더를 스캔하고 검증 결과를 dict로 반환.

    반환:
      {
        "images": [paths], "labels": [paths],
        "n_img": int, "n_lbl": int,
        "missing_label": [img stem...],   # 라벨 없는 이미지
        "empty_label":   [stem...],       # 빈 txt (배경 이미지일 수도 있음 — 경고만)
        "orphan_label":  [stem...],       # 이미지 없는 라벨
        "class_ids":     sorted set,      # txt에서 발견된 클래스 ID
        "max_class_id":  int or None,
        "bad_lines":     [(file, lineno, reason)...],  # 형식 오류
        "ok": bool, "messages": [str...]
      }
    """
    messages = []
    result = {
        "images": [], "labels": [], "n_img": 0, "n_lbl": 0,
        "missing_label": [], "empty_label": [], "orphan_label": [],
        "class_ids": [], "max_class_id": None, "bad_lines": [],
        "ok": False, "messages": messages,
    }

    if not img_dir or not os.path.isdir(img_dir):
        messages.append(f"[오류] 이미지 폴더가 없습니다: {img_dir}")
        return result
    if not lbl_dir or not os.path.isdir(lbl_dir):
        messages.append(f"[오류] 라벨 폴더가 없습니다: {lbl_dir}")
        return result

    images = _list_images(img_dir, img_ext)
    all_labels = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))
    result["images"] = images
    result["labels"] = all_labels
    result["n_img"] = len(images)
    result["n_lbl"] = len(all_labels)

    if not images:
        messages.append(f"[오류] .{img_ext} 이미지가 없습니다: {img_dir}")
        return result

    img_stems = {os.path.splitext(os.path.basename(p))[0] for p in images}
    lbl_stems = {os.path.splitext(os.path.basename(p))[0] for p in all_labels}

    class_ids = set()

    for img in images:
        lp = _label_for(img, lbl_dir)
        stem = os.path.splitext(os.path.basename(img))[0]
        if not os.path.isfile(lp):
            result["missing_label"].append(stem)
            continue
        # 라벨 파싱 + 검증
        try:
            with open(lp, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except Exception as e:
            result["bad_lines"].append((os.path.basename(lp), 0, f"읽기 실패: {e}"))
            continue
        if not lines:
            result["empty_label"].append(stem)
            continue
        for i, ln in enumerate(lines, 1):
            parts = ln.split()
            # Detection: class cx cy w h  (5개)
            if len(parts) != 5:
                result["bad_lines"].append(
                    (os.path.basename(lp), i, f"열 개수 {len(parts)} (5개 필요)"))
                continue
            try:
                cid = int(float(parts[0]))
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                result["bad_lines"].append(
                    (os.path.basename(lp), i, "숫자 변환 실패"))
                continue
            class_ids.add(cid)
            if any(c < 0 or c > 1 for c in coords):
                result["bad_lines"].append(
                    (os.path.basename(lp), i, "좌표가 0~1 범위를 벗어남 (정규화 필요)"))

    # 이미지 없는 라벨(고아)
    result["orphan_label"] = sorted(lbl_stems - img_stems)

    result["class_ids"] = sorted(class_ids)
    result["max_class_id"] = max(class_ids) if class_ids else None

    # 요약 메시지
    messages.append(f"이미지 {result['n_img']}개 / 라벨(txt) {result['n_lbl']}개")
    if result["class_ids"]:
        messages.append(f"발견된 클래스 ID: {result['class_ids']}  (최대 {result['max_class_id']})")
    if result["missing_label"]:
        messages.append(f"[경고] 라벨 없는 이미지 {len(result['missing_label'])}개 "
                        f"(예: {result['missing_label'][:5]})")
    if result["empty_label"]:
        messages.append(f"[정보] 빈 라벨 {len(result['empty_label'])}개 "
                        f"(배경 이미지로 처리됨)")
    if result["orphan_label"]:
        messages.append(f"[경고] 이미지 없는 라벨 {len(result['orphan_label'])}개 "
                        f"(예: {result['orphan_label'][:5]})")
    if result["bad_lines"]:
        messages.append(f"[오류] 형식 오류 {len(result['bad_lines'])}줄 발견:")
        for fn, ln, why in result["bad_lines"][:10]:
            messages.append(f"    {fn}:{ln}  — {why}")
        if len(result["bad_lines"]) > 10:
            messages.append(f"    ... 외 {len(result['bad_lines'])-10}줄")

    # ok 판정: 형식 오류 없고, 라벨 달린 이미지가 1개 이상
    n_labeled = result["n_img"] - len(result["missing_label"])
    result["ok"] = (not result["bad_lines"]) and (n_labeled > 0)
    if result["ok"]:
        messages.append(f"✔ 검증 통과 (학습 가능 이미지 {n_labeled}개)")
    else:
        messages.append("✘ 검증 실패 — 위 문제를 해결하세요")
    return result


# ──────────────────────────────────────────────────────────────────
#  data.yaml 생성
# ──────────────────────────────────────────────────────────────────
def _copy_pair(img_path, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    """이미지 + 짝 라벨(txt)을 대상 폴더로 복사"""
    shutil.copy(img_path, dst_img_dir)
    lp = _label_for(img_path, src_lbl_dir)
    if os.path.isfile(lp):
        shutil.copy(lp, dst_lbl_dir)


def build_yaml(out_dir, class_names,
               split_mode="manual",
               train_img_dir=None, train_lbl_dir=None,
               val_img_dir=None,   val_lbl_dir=None,
               img_ext="png", holdout_ratio=0.2,
               aug=None, seed=42, log=print):
    """
    Ultralytics용 data.yaml 생성.

    split_mode:
      - "manual" : train/val 폴더가 이미 분리됨 → 경로만 yaml에 기록
      - "holdout": train 폴더 데이터를 랜덤 분할 → out_dir/dataset/{train,val}로 복사

    반환: data.yaml 절대경로
    """
    import yaml
    os.makedirs(out_dir, exist_ok=True)
    yaml_path = os.path.join(out_dir, "data.yaml")
    names = {i: n for i, n in enumerate(class_names)}

    if split_mode == "manual":
        content = {
            "train": os.path.abspath(train_img_dir),
            "val":   os.path.abspath(val_img_dir),
            "nc":    len(class_names),
            "names": names,
        }
        log(f"  [yaml] manual — train: {content['train']}")
        log(f"  [yaml] manual — val:   {content['val']}")

    elif split_mode == "holdout":
        ds = os.path.join(out_dir, "dataset")
        ti = os.path.join(ds, "train", "images"); tl = os.path.join(ds, "train", "labels")
        vi = os.path.join(ds, "val",   "images"); vl = os.path.join(ds, "val",   "labels")
        for d in (ti, tl, vi, vl):
            os.makedirs(d, exist_ok=True)

        imgs = _list_images(train_img_dir, img_ext)
        random.Random(seed).shuffle(imgs)
        n_val = max(1, int(len(imgs) * holdout_ratio))
        val_set, train_set = imgs[:n_val], imgs[n_val:]

        for p in train_set:
            _copy_pair(p, train_lbl_dir, ti, tl)
        for p in val_set:
            _copy_pair(p, train_lbl_dir, vi, vl)

        log(f"  [yaml] hold-out — Train {len(train_set)} / Val {len(val_set)} "
            f"(ratio={holdout_ratio})")
        content = {
            "train": os.path.abspath(ti),
            "val":   os.path.abspath(vi),
            "nc":    len(class_names),
            "names": names,
        }
    else:
        raise ValueError(f"알 수 없는 split_mode: {split_mode}")

    # Augmentation 키 병합 (있을 때만)
    if aug:
        content.update(aug)

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, allow_unicode=True, sort_keys=False)
    log(f"  [yaml] 저장: {yaml_path}")
    return yaml_path


# ──────────────────────────────────────────────────────────────────
#  학습
# ──────────────────────────────────────────────────────────────────
def _extract_metrics(trainer):
    """
    trainer에서 epoch별 지표를 안전하게 추출.
    Ultralytics 버전마다 키가 조금씩 달라 여러 경로를 시도.
    반환: dict(box_loss, cls_loss, dfl_loss, mAP50, mAP50_95, precision, recall)
    """
    out = {"box_loss": None, "cls_loss": None, "dfl_loss": None,
           "mAP50": None, "mAP50_95": None, "precision": None, "recall": None}

    # 손실 — trainer.loss_items / trainer.tloss
    label_loss = getattr(trainer, "loss_names", None)  # ('box_loss','cls_loss','dfl_loss')
    loss_items = getattr(trainer, "tloss", None)
    if loss_items is None:
        loss_items = getattr(trainer, "loss_items", None)
    try:
        if label_loss is not None and loss_items is not None:
            vals = [float(v) for v in loss_items]
            for name, v in zip(label_loss, vals):
                if "box" in name: out["box_loss"] = v
                elif "cls" in name: out["cls_loss"] = v
                elif "dfl" in name: out["dfl_loss"] = v
    except Exception:
        pass

    # 검증 지표 — trainer.metrics dict
    metrics = getattr(trainer, "metrics", None)
    if isinstance(metrics, dict):
        def pick(*keys):
            for k in keys:
                if k in metrics:
                    try:
                        return float(metrics[k])
                    except Exception:
                        pass
            return None
        out["mAP50"]     = pick("metrics/mAP50(B)", "metrics/mAP50", "mAP50")
        out["mAP50_95"]  = pick("metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95")
        out["precision"] = pick("metrics/precision(B)", "metrics/precision")
        out["recall"]    = pick("metrics/recall(B)", "metrics/recall")
    return out


def train(config, on_epoch=None, on_log=None, should_stop=None):
    """
    Ultralytics YOLO Detection 학습.

    config 키:
      model      : "yolov8n.pt" 등
      data_yaml  : data.yaml 경로
      epochs, batch, imgsz, lr0, optimizer, patience
      project, name  : 결과 저장 경로 (project/name/)
    콜백:
      on_epoch(epoch, total, metrics_dict)
      on_log(str)
      should_stop() -> bool
    반환: dict(success, best_pt, save_dir, error)
    """
    def log(msg):
        if on_log: on_log(msg)

    # GUI 내장 실행이므로 wandb 등 외부 로거 자동 연동을 끈다 (예기치 않은 로그인/업로드 방지)
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("YOLO_VERBOSE", "True")

    try:
        from ultralytics import YOLO
        from ultralytics import settings as yolo_settings
        import torch
    except ImportError as e:
        return {"success": False, "error":
                f"ultralytics/torch 미설치: {e}\n→ pip install ultralytics"}

    # Ultralytics 내장 로거(wandb/comet 등) 비활성화
    try:
        yolo_settings.update({"wandb": False, "comet": False, "tensorboard": False,
                              "mlflow": False, "dvc": False, "neptune": False})
    except Exception:
        pass

    model_name = config.get("model", "yolov8n.pt")
    data_yaml  = config.get("data_yaml")
    epochs     = int(config.get("epochs", 100))
    device     = 0 if torch.cuda.is_available() else "cpu"
    # project를 절대경로로 만들어 Ultralytics가 runs/detect/ 아래로 중첩 저장하지 않게 함
    project = config.get("project", "YOLO_Project")
    project = os.path.abspath(project)

    log(f"[학습 시작] 디바이스: {'GPU(cuda:0)' if device == 0 else 'CPU'}")
    log(f"  모델: {model_name} | Epochs: {epochs} | "
        f"Batch: {config.get('batch', 16)} | imgsz: {config.get('imgsz', 640)}")
    log(f"  data.yaml: {data_yaml}")

    try:
        model = YOLO(model_name)
    except Exception as e:
        return {"success": False, "error": f"모델 로드 실패 ({model_name}): {e}"}

    # epoch별 mAP를 추적해 best를 기록 (K-Fold 집계용)
    best_map50 = {"v": 0.0}
    best_map5095 = {"v": 0.0}

    # ── epoch 종료 콜백: 지표 추출 → GUI로 전달 + 중단 처리 ──────────
    def _on_fit_epoch_end(trainer):
        ep = int(getattr(trainer, "epoch", 0)) + 1   # 0-indexed → 1-indexed
        metrics = _extract_metrics(trainer)
        if metrics.get("mAP50") is not None and metrics["mAP50"] > best_map50["v"]:
            best_map50["v"] = metrics["mAP50"]
            best_map5095["v"] = metrics.get("mAP50_95") or best_map5095["v"]
        if on_epoch:
            on_epoch(ep, epochs, metrics)
        if should_stop and should_stop():
            log("  [중단 요청] 다음 epoch에서 학습을 멈춥니다...")
            trainer.stop = True            # Ultralytics가 인식하는 중단 플래그
            try:
                trainer.stop_training = True
            except Exception:
                pass

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

    train_kwargs = dict(
        data=data_yaml,
        epochs=epochs,
        batch=int(config.get("batch", 16)),
        imgsz=int(config.get("imgsz", 640)),
        device=device,
        project=project,
        name=config.get("name", "train"),
        exist_ok=True,
        optimizer=config.get("optimizer", "auto"),
        lr0=float(config.get("lr0", 0.01)),
        patience=int(config.get("patience", 50)),
        workers=int(config.get("workers", 4)),
        plots=True,
        verbose=True,
    )
    # augmentation 등 추가 키 병합 (있을 때만 전달)
    for k in ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
              "shear", "perspective", "flipud", "fliplr", "mosaic",
              "mixup", "copy_paste", "erasing"):
        if k in config and config[k] is not None:
            train_kwargs[k] = config[k]

    try:
        results = model.train(**train_kwargs)
    except Exception as e:
        import traceback
        return {"success": False, "error": f"{e}\n{traceback.format_exc()}"}

    save_dir = str(getattr(results, "save_dir", "") or
                   getattr(getattr(model, "trainer", None), "save_dir", ""))
    best_pt = os.path.join(save_dir, "weights", "best.pt") if save_dir else ""
    log(f"\n✔ 학습 완료")
    if save_dir:
        log(f"  결과 폴더: {save_dir}")
    if best_pt and os.path.isfile(best_pt):
        log(f"  Best 모델: {best_pt}")

    return {"success": True, "best_pt": best_pt, "save_dir": save_dir, "error": None,
            "best_map50": best_map50["v"], "best_map5095": best_map5095["v"]}


# ──────────────────────────────────────────────────────────────────
#  K-Fold
# ──────────────────────────────────────────────────────────────────
def build_kfold_yamls(out_dir, class_names, train_img_dir, train_lbl_dir,
                      k=5, img_ext="png", aug=None, seed=42, log=print):
    """
    Train 데이터를 K등분해 fold별 dataset 폴더 + data.yaml 생성.
    fold i 는 i번째 조각을 val, 나머지를 train 으로 사용.

    반환: [yaml_path_fold1, ..., yaml_path_foldK]
    """
    import yaml
    os.makedirs(out_dir, exist_ok=True)
    names = {i: n for i, n in enumerate(class_names)}

    imgs = _list_images(train_img_dir, img_ext)
    if len(imgs) < k:
        raise ValueError(f"이미지 {len(imgs)}개 < Fold 수 {k} — K를 줄이거나 데이터를 늘리세요")
    random.Random(seed).shuffle(imgs)

    # K등분 (앞쪽 fold가 1개 더 가질 수 있음)
    folds = [imgs[i::k] for i in range(k)]

    yaml_paths = []
    for i in range(k):
        val_set = folds[i]
        train_set = [p for j in range(k) if j != i for p in folds[j]]

        ds = os.path.join(out_dir, f"fold_{i+1}", "dataset")
        ti = os.path.join(ds, "train", "images"); tl = os.path.join(ds, "train", "labels")
        vi = os.path.join(ds, "val",   "images"); vl = os.path.join(ds, "val",   "labels")
        for d in (ti, tl, vi, vl):
            os.makedirs(d, exist_ok=True)

        for p in train_set:
            _copy_pair(p, train_lbl_dir, ti, tl)
        for p in val_set:
            _copy_pair(p, train_lbl_dir, vi, vl)

        content = {
            "train": os.path.abspath(ti),
            "val":   os.path.abspath(vi),
            "nc":    len(class_names),
            "names": names,
        }
        if aug:
            content.update(aug)

        yp = os.path.join(out_dir, f"fold_{i+1}", "data.yaml")
        with open(yp, "w", encoding="utf-8") as f:
            yaml.dump(content, f, allow_unicode=True, sort_keys=False)
        yaml_paths.append(yp)
        log(f"  [K-Fold] Fold {i+1}/{k}: Train {len(train_set)} / Val {len(val_set)}")

    return yaml_paths


def build_full_yaml(out_dir, class_names, train_img_dir, train_lbl_dir,
                    img_ext="png", aug=None, log=print):
    """전체 데이터를 train=val로 쓰는 data.yaml (K-Fold 후 Full 재학습용)"""
    import yaml
    os.makedirs(out_dir, exist_ok=True)
    names = {i: n for i, n in enumerate(class_names)}
    content = {
        "train": os.path.abspath(train_img_dir),
        "val":   os.path.abspath(train_img_dir),  # Full 학습: 전체를 val로도 사용
        "nc":    len(class_names),
        "names": names,
    }
    if aug:
        content.update(aug)
    yp = os.path.join(out_dir, "data_full.yaml")
    with open(yp, "w", encoding="utf-8") as f:
        yaml.dump(content, f, allow_unicode=True, sort_keys=False)
    log(f"  [Full] data.yaml 저장: {yp}")
    return yp


def train_kfold(config, on_epoch=None, on_log=None, should_stop=None, on_fold=None):
    """
    K-Fold 학습: fold마다 train() 호출 → 평균±std 집계 → Full 데이터 재학습.

    config 키 (train()과 동일) + kfold_k, train_img_dir, train_lbl_dir, img_ext, class_names, aug
    추가 콜백:
      on_fold(fold_idx, total_folds)  # fold_idx=0,total=0 → Full 학습 시작 신호
    반환: dict(success, best_pt, save_dir, fold_results, mean_map50, std_map50, error)
    """
    import statistics
    def log(msg):
        if on_log: on_log(msg)

    k          = int(config.get("kfold_k", 5))
    project    = os.path.abspath(config.get("project", "YOLO_Project"))
    class_names = config["class_names"]
    aug        = config.get("aug")

    log(f"[K-Fold 시작] K={k}")
    try:
        yaml_paths = build_kfold_yamls(
            out_dir=project, class_names=class_names,
            train_img_dir=config["train_img_dir"], train_lbl_dir=config["train_lbl_dir"],
            k=k, img_ext=config.get("img_ext", "png"), aug=aug, log=log)
    except Exception as e:
        return {"success": False, "error": f"K-Fold 분할 실패: {e}"}

    fold_results = []
    for i, yp in enumerate(yaml_paths):
        if should_stop and should_stop():
            log("[중단] K-Fold 중지")
            break
        log(f"\n{'='*46}\n── Fold {i+1}/{k} 학습 ──")
        if on_fold:
            on_fold(i + 1, k)   # 그래프 초기화 신호
        fold_cfg = dict(config)
        fold_cfg["data_yaml"] = yp
        fold_cfg["name"] = f"fold_{i+1}"
        res = train(fold_cfg, on_epoch=on_epoch, on_log=on_log, should_stop=should_stop)
        if not res.get("success"):
            return {"success": False, "error": f"Fold {i+1} 학습 실패: {res.get('error')}"}
        fold_results.append({
            "fold": i + 1,
            "best_map50": round(res.get("best_map50", 0.0), 6),
            "best_map5095": round(res.get("best_map5095", 0.0), 6),
            "best_pt": res.get("best_pt", ""),
        })

    # ── 집계 ──────────────────────────────────────────────────────
    summary = {"success": True, "fold_results": fold_results, "error": None}
    if fold_results:
        m50 = [r["best_map50"] for r in fold_results]
        m5095 = [r["best_map5095"] for r in fold_results]
        summary["mean_map50"] = round(statistics.mean(m50), 6)
        summary["std_map50"]  = round(statistics.pstdev(m50), 6) if len(m50) > 1 else 0.0
        summary["mean_map5095"] = round(statistics.mean(m5095), 6)
        summary["std_map5095"]  = round(statistics.pstdev(m5095), 6) if len(m5095) > 1 else 0.0
        log(f"\n{'='*46}\n[K-Fold 결과 요약]")
        for r in fold_results:
            log(f"  Fold {r['fold']}: mAP50={r['best_map50']:.4f}  mAP50-95={r['best_map5095']:.4f}")
        log(f"  평균 mAP50    : {summary['mean_map50']:.4f} ± {summary['std_map50']:.4f}")
        log(f"  평균 mAP50-95 : {summary['mean_map5095']:.4f} ± {summary['std_map5095']:.4f}")

        # JSON 저장
        import json
        with open(os.path.join(project, "kfold_summary.json"), "w", encoding="utf-8") as f:
            json.dump({kk: vv for kk, vv in summary.items() if kk != "success"},
                      f, indent=2, ensure_ascii=False)

    # ── Full 데이터 재학습 (최종 모델) ────────────────────────────
    if config.get("full_retrain", True) and not (should_stop and should_stop()):
        log(f"\n{'='*46}\n[Full Data 학습]  전체 데이터로 최종 모델 생성")
        if on_fold:
            on_fold(0, 0)   # Full 학습 신호 (그래프 초기화)
        try:
            full_yaml = build_full_yaml(
                out_dir=project, class_names=class_names,
                train_img_dir=config["train_img_dir"], train_lbl_dir=config["train_lbl_dir"],
                img_ext=config.get("img_ext", "png"), aug=aug, log=log)
            full_cfg = dict(config)
            full_cfg["data_yaml"] = full_yaml
            full_cfg["name"] = "full"
            res_full = train(full_cfg, on_epoch=on_epoch, on_log=on_log, should_stop=should_stop)
            if res_full.get("success"):
                summary["best_pt"]  = res_full.get("best_pt", "")
                summary["save_dir"] = res_full.get("save_dir", "")
                log(f"  ✔ Full 최종 모델: {summary['best_pt']}")
        except Exception as e:
            log(f"  [Full 학습 오류] {e}")

    # Full 학습을 안 했거나 실패하면 마지막 fold best를 대표 모델로
    if "best_pt" not in summary and fold_results:
        summary["best_pt"]  = fold_results[-1]["best_pt"]
        summary["save_dir"] = os.path.dirname(os.path.dirname(fold_results[-1]["best_pt"]))

    return summary
