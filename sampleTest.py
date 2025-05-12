import cv2
from ultralytics import YOLO
import torch, timm, json, time
from torchvision import transforms
from pathlib import Path
from PIL import Image

CONF_THRES = 0.25
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = Path("captures")
OUT_DIR.mkdir(exist_ok=True)

detector = YOLO("yolov8m.pt")
cls_model = timm.create_model(
    "vit_base_patch16_224_in21k",
    pretrained=False,
    num_classes=10000
).to(DEVICE)
cls_model.load_state_dict(
    torch.hub.load_state_dict_from_url(
        "https://storage.googleapis.com/imagenet‑weights/iNat21_ViT_B16.pt"
    )
)
cls_model.eval()

im_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

with open("ca_species.txt") as f:
    CA_SPECIES = {s.strip().lower() for s in f}


def in_ca(label: str) -> bool:
    return label.lower() in CA_SPECIES


cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Cannot open camera"

seen = set()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    for r in detector.predict(frame, conf=CONF_THRES, verbose=False):
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            crop = frame[y1:y2, x1:x2]

            img = Image.fromarray(crop[:, :, ::-1])
            with torch.no_grad():
                pred = cls_model(im_tf(img).unsqueeze(0).to(DEVICE))[0].softmax(0)
            top5 = pred.topk(5)
            best_idx = top5.indices[0].item()
            best_prob = top5.values[0].item()
            label = cls_model.pretrained_cfg["label_names"][best_idx]

            if in_ca(label):
                seen.add(label)
                ts = time.strftime("%Y%m%d_%H%M%S")
                name = f"{ts}_{label}.jpg"
                cv2.imwrite(str(OUT_DIR / name), crop)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {best_prob:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

    cv2.imshow("California Wildlife", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Unique CA species observed:", len(seen))
