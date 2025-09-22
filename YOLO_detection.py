import os
from datetime import datetime
from typing import Union

import cv2
import torch
from ultralytics import YOLO


def _select_device(requested: str | None = None) -> str:
    """自动选择设备（如果要求），或“自动”。
    更喜欢CUDA，然后是MPS（Apple），Else CPU。"""
    if requested and requested.lower() not in {"auto", ""}:
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class YOLODetector:
    def __init__(self, model_path: str = 'yolo11n.pt', device: str | None = 'auto', save_txt: bool = False):
        """初始化Yolov11（超级分析）检测器。

        model_path：模型权重文件（可以是yolo11n.pt或经过自定义训练的.pt）
        设备：“ CUDA”，“ CPU”，“ MPS”或“自动”
        save_txt：是否还要将检测结果（每帧）导出到图像旁边的.txt
        """
        self.device = _select_device(device)
        # Load model
        self.model: YOLO = YOLO(model_path)
        self.save_txt = save_txt

    def detect_and_save(self, source: Union[int, str] = 0, save_dir: str = 'results', conf: float = 0.5, imgsz: int | list[int] | None = None):
        os.makedirs(save_dir, exist_ok=True)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源： {source}")

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run prediction (Ultralytics YOLO automatically handles device placement internally)
            if imgsz is not None:
                results = self.model.predict(frame, imgsz=imgsz, conf=conf, device=self.device, verbose=False)
            else:
                # Let model use its default image size
                results = self.model.predict(frame, conf=conf, device=self.device, verbose=False)
            # results is a list (batch). We passed single frame so take first.
            result = results[0]

            annotated_frame = result.plot()  # Draw boxes, labels

            cv2.imshow('YOLOv11 Detection', annotated_frame)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = f'frame_{frame_id}_{timestamp}'
            img_path = os.path.join(save_dir, base_name + '.jpg')
            cv2.imwrite(img_path, annotated_frame)

            if self.save_txt:
                # Write YOLO format txt: class x_center y_center w h (normalized)
                txt_path = os.path.join(save_dir, base_name + '.txt')
                h, w = result.orig_shape
                with open(txt_path, 'w', encoding='utf-8') as f:
                    for box in result.boxes:  # type: ignore[attr-defined]
                        cls_id = int(box.cls.item())
                        xyxy = box.xyxy[0].tolist()  # [x1,y1,x2,y2]
                        x1, y1, x2, y2 = xyxy
                        x_c = (x1 + x2) / 2 / w
                        y_c = (y1 + y2) / 2 / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

            frame_id += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


