from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from pathlib import Path
Dst = Path(r"C:\Users\JJL\Downloads\KolektorSDD2\abn")
datamodule = Folder(
    name="kolektor",
    root=Dst,                  # 영문 경로만 가능 (한글 X)
    normal_dir=r"C:\Users\JJL\Downloads\KolektorSDD2\abn\train_good",   # 학습: 정상 300장
    normal_test_dir=r"C:\Users\JJL\Downloads\KolektorSDD2\abn\test_good",  # 평가: 정상 300장 (따로 지정하면 학습 폴더를 쪼개지 않음)
    abnormal_dir="abnorm",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,             # Windows에서 데이터 로더 에러 방지
)

model = Patchcore(
    backbone="wide_resnet50_2",          # 사전학습 CNN
    layers=["layer2", "layer3"],         # 중간층 특징 사용
    coreset_sampling_ratio=0.1,          # 메모리 뱅크에 남길 비율
)


SaveDir = Path(r"C:\Users\JJL\Desktop\Python Code Develop\Segmentation Model maker\results")
engine = Engine(default_root_dir=SaveDir)   # 실행 위치와 상관없이 항상 여기에 저장
engine.fit(model=model, datamodule=datamodule)

# 고정 이름으로 저장 (다시 학습하면 덮어씀, Anomaly_View.py 가 이 파일을 읽음)
CkptPath = SaveDir / "patchcore_kolektor.ckpt"
engine.trainer.save_checkpoint(CkptPath)
print(f"모델 저장: {CkptPath}")

engine.test(model=model, datamodule=datamodule)   # AUROC 등 평가
