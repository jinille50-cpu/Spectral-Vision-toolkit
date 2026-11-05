import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision.io import read_image, ImageReadMode
from anomalib.models import Patchcore
import File_Util as Fu

# Anomalib.py 에서 학습 후 저장한 체크포인트 (메모리 뱅크 + 임계값 포함)
Ckpt = Path(r"C:\Users\JJL\Desktop\Python Code Develop\results\Patchcore\model.ckpt")
ImgDir = r"C:\Users\JJL\Downloads\KolektorSDD2\abn"
Exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_model(ckpt):
    """체크포인트를 1번만 불러와서 계속 재사용"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Patchcore.load_from_checkpoint(ckpt)
    model.eval().to(device)
    return model, device


def predict(img_path, model, device):
    """이미지 1장 판정 -> (원본, 점수, 불량여부, 히트맵, 마스크) 반환"""
    # 0~255 정수 -> 0~1 실수, (채널,H,W) -> (1,채널,H,W)  (흑백도 3채널로 읽음)
    x = read_image(str(img_path), ImageReadMode.RGB).float().div(255).unsqueeze(0).to(device)
    with torch.no_grad():  # 추론만 하므로 기울기 계산 끄기 (빠르고 메모리 절약)
        # 전처리(크기 조정·정규화) -> PatchCore -> 점수 정규화·임계값 판정
        # model(x) 로 바로 넣으면 학습 때와 다른 크기 조정 방식이 쓰여 점수가 약간 달라짐 -> 학습 때 전처리를 직접 적용
        out = model.post_processor(model.model(model.pre_processor.transform(x)))

    score = float(out.pred_score[0])
    label = bool(out.pred_label[0])
    # 히트맵·마스크는 모델 입력 크기(256x256) -> 원본 이미지 크기로 맞춤 (MATLAB imresize)
    H, W = x.shape[-2:]
    amap = F.interpolate(out.anomaly_map.reshape(1, 1, *out.anomaly_map.shape[-2:]).float(),
                         size=(H, W), mode="bilinear")[0, 0].cpu().numpy()
    mask = F.interpolate(out.pred_mask.reshape(1, 1, *out.pred_mask.shape[-2:]).float(),
                         size=(H, W), mode="nearest")[0, 0].cpu().numpy()
    im = x[0].permute(1, 2, 0).cpu().numpy()  # (채널,H,W) -> (H,W,채널) 그림용
    return im, score, label, amap, mask


def draw(fig, ax, img_path, idx, n, im, score, label, amap, mask):
    """원본 | 원본 + 히트맵 + 불량 영역 외곽선"""
    for a in ax:
        a.clear()  # 이전 그림 지우고 같은 창에 다시 그림
        a.axis("off")
    ax[0].imshow(im)
    ax[0].set_title("원본")

    ax[1].imshow(im)
    ax[1].imshow(amap, cmap="jet", alpha=0.4, vmin=0, vmax=1)  # 반투명 히트맵 겹치기
    if mask.any():
        ax[1].contour(mask, levels=[0.5], colors="white", linewidths=1)  # 불량 영역 외곽선
    ax[1].set_title(f"{'불량' if label else '정상'}  (점수 {score:.3f})",
                    color="red" if label else "green")

    fig.suptitle(f"[{idx + 1}/{n}]  {Path(img_path).name}     ← → 이동")
    fig.canvas.draw_idle()  # 창 갱신


# 이 파일을 직접 실행할 때만 아래가 돈다 (다른 코드에서 import 할 때는 안 돎)
if __name__ == "__main__":
    plt.rcParams["font.family"] = "Malgun Gothic"  # 한글 제목 깨짐 방지
    plt.rcParams["keymap.back"] = []     # ← → 가 matplotlib 기본 단축키(뒤로/앞으로)와 겹치지 않게
    plt.rcParams["keymap.forward"] = []

    model, device = load_model(Ckpt)

    folder = Fu.Uigetdir(ImgDir, "이미지 폴더 선택")
    imgs = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in Exts) if folder else []
    if not imgs:
        raise SystemExit("이미지가 없습니다.")

    fig, ax = plt.subplots(1, 2, figsize=(8, 8))
    state = {"i": 0}  # 현재 이미지 번호 (함수 안에서 바꾸려고 dict 에 담음)

    def show(i):
        state["i"] = i % len(imgs)  # %: 끝에서 다음 누르면 처음으로
        p = imgs[state["i"]]
        draw(fig, ax, p, state["i"], len(imgs), *predict(p, model, device))  # *: 반환값 묶음을 풀어서 인자로 전달

    def on_key(event):
        if event.key == "right":
            show(state["i"] + 1)
        elif event.key == "left":
            show(state["i"] - 1)

    fig.canvas.mpl_connect("key_press_event", on_key)  # 키 누를 때마다 on_key 실행
    show(0)
    plt.tight_layout()
    plt.show()
