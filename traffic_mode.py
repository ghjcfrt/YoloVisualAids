"""交通灯颜色测试模式的实现。

提供 run(args) 给 test.py 调用，支持：
- 图片/目录/摄像头
- 手动 ROI（交互/参数）或 YOLO 自动识别裁剪
"""

from __future__ import annotations

import os
from typing import Optional, Tuple


def run(args) -> None:  # noqa: C901 - 控制流较多但清晰
    import cv2

    from color_detction import detect_traffic_light_color
    from roi_utils import clamp_roi, iter_images_from_dir, pick_roi_interactive
    from yolo_utils import YOLOAutoDetector, YOLOOpts, parse_img_size

    win = getattr(args, 'win', 'traffic-test')
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def draw_and_report(frame, roi_rect: Tuple[int, int, int, int]):
        x, y, w, h = roi_rect
        x, y, w, h = clamp_roi(x, y, w, h, frame.shape[1], frame.shape[0])
        roi = frame[y : y + h, x : x + w]
        color = detect_traffic_light_color(roi)
        out = frame.copy()
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            out,
            f"color: {color}",
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255) if color == "yellow" else ((0, 255, 0) if color == "green" else (0, 0, 255)),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(win, out)
        return color

    # YOLO 自动检测（按需）
    detector: Optional[YOLOAutoDetector] = None
    if getattr(args, 'auto', False):
        opts = YOLOOpts(
            model_path=getattr(args, 'model', 'yolo11n.pt'),
            conf=float(getattr(args, 'conf', 0.5)),
            img_size=parse_img_size(getattr(args, 'img_size', '')),
            device=getattr(args, 'device', 'auto'),
            class_id=int(getattr(args, 'class_id', 9)),
            first_only=bool(getattr(args, 'first', False)),
        )
        detector = YOLOAutoDetector(opts)

    roi_rect: Optional[Tuple[int, int, int, int]] = tuple(args.roi) if getattr(args, 'roi', None) else None

    # 单张图片
    if getattr(args, 'image', None):
        path = args.image
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"无法读取图片: {path}")
        if detector is not None:  # 自动
            boxes = detector.detect(img)
            out = img.copy()
            if not boxes:
                print(f"{os.path.basename(path)}: 未检测到交通灯 (class-id={getattr(args,'class_id',9)})")
            for (x1, y1, x2, y2, conf) in boxes:
                roi = img[y1:y2, x1:x2]
                color = detect_traffic_light_color(roi)
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(out, f"{color} {conf:.2f}", (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 200, 255) if color == "yellow" else ((0, 255, 0) if color == "green" else (0, 0, 255)), 2, cv2.LINE_AA)
                if getattr(args, 'save_crops', ''):
                    os.makedirs(args.save_crops, exist_ok=True)
                    base = os.path.splitext(os.path.basename(path))[0]
                    crop_name = f"{base}_{x1}_{y1}_{x2}_{y2}_{color}.jpg"
                    cv2.imwrite(os.path.join(args.save_crops, crop_name), roi)
                print(f"{os.path.basename(path)}: {color} (conf={conf:.2f}) box=({x1},{y1},{x2},{y2})")
            cv2.imshow(win, out)
            print("按任意键退出… q 也可")
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k in (ord('q'), 27, 13):
                    break
        else:  # 手动
            if roi_rect is None:
                roi_rect = pick_roi_interactive(win, img)
                if roi_rect is None:
                    print("未选择 ROI，退出。")
                    return
            color = draw_and_report(img, roi_rect)
            print(f"{path}: {color}")
            print("按任意键退出… q 也可")
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k in (ord('q'), 27, 13):
                    break
        cv2.destroyAllWindows()
        return

    # 目录
    if getattr(args, 'dir', None):
        dirp = args.dir
        if not os.path.isdir(dirp):
            raise NotADirectoryError(dirp)
        images = iter_images_from_dir(dirp)
        if not images:
            print("目录中未找到图片。支持: .jpg .jpeg .png .bmp .webp")
            return
        idx = 0
        while idx < len(images):
            path = images[idx]
            img = cv2.imread(path)
            if img is None:
                print(f"读取失败，跳过: {path}")
                idx += 1
                continue
            if detector is not None:  # 自动
                boxes = detector.detect(img)
                out = img.copy()
                results = []
                for (x1, y1, x2, y2, conf) in boxes:
                    roi = img[y1:y2, x1:x2]
                    color = detect_traffic_light_color(roi)
                    results.append((color, conf, (x1, y1, x2, y2)))
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(out, f"{color} {conf:.2f}", (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 200, 255) if color == "yellow" else ((0, 255, 0) if color == "green" else (0, 0, 255)), 2, cv2.LINE_AA)
                    if getattr(args, 'save_crops', ''):
                        os.makedirs(args.save_crops, exist_ok=True)
                        base = os.path.splitext(os.path.basename(path))[0]
                        crop_name = f"{base}_{x1}_{y1}_{x2}_{y2}_{color}.jpg"
                        cv2.imwrite(os.path.join(args.save_crops, crop_name), roi)
                if not boxes:
                    print(f"{os.path.basename(path)}: 未检测到交通灯 (class-id={getattr(args,'class_id',9)})")
                else:
                    for color, conf, (x1, y1, x2, y2) in results:
                        print(f"{os.path.basename(path)}: {color} (conf={conf:.2f}) box=({x1},{y1},{x2},{y2})")
                cv2.imshow(win, out)
                k = cv2.waitKey(0) & 0xFF
            else:  # 手动
                local_roi = roi_rect
                if local_roi is None:
                    local_roi = pick_roi_interactive(win, img)
                    if local_roi is None:
                        print("未选择 ROI，退出。")
                        break
                color = draw_and_report(img, local_roi)
                print(f"{os.path.basename(path)}: {color}")
                k = cv2.waitKey(0) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k in (ord('n'), 13):
                idx += 1
            elif k in (ord('r'),):
                roi_rect = None
            else:
                idx += 1
        cv2.destroyAllWindows()
        return

    # 摄像头
    if getattr(args, 'cam', None) is not None:
        cap = cv2.VideoCapture(int(args.cam))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {args.cam}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("读取摄像头帧失败")
                    break
                if detector is not None:
                    boxes = detector.detect(frame)
                    out = frame.copy()
                    for (x1, y1, x2, y2, conf) in boxes:
                        roi = frame[y1:y2, x1:x2]
                        color = detect_traffic_light_color(roi)
                        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(out, f"{color} {conf:.2f}", (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 200, 255) if color == "yellow" else ((0, 255, 0) if color == "green" else (0, 0, 255)), 2, cv2.LINE_AA)
                        if getattr(args, 'save_crops', ''):
                            try:
                                os.makedirs(args.save_crops, exist_ok=True)
                                crop_name = f"cam_{x1}_{y1}_{x2}_{y2}_{color}.jpg"
                                cv2.imwrite(os.path.join(args.save_crops, crop_name), roi)
                            except Exception:
                                pass
                    cv2.imshow(win, out)
                else:
                    if roi_rect is None:
                        roi_rect = pick_roi_interactive(win, frame)
                        if roi_rect is None:
                            print("未选择 ROI，退出。")
                            break
                    draw_and_report(frame, roi_rect)
                k = cv2.waitKey(max(1, int(getattr(args, 'delay', 1)))) & 0xFF
                if k in (ord('q'), 27):
                    break
                elif k in (ord('r'),):
                    roi_rect = None
        finally:
            cap.release()
        cv2.destroyAllWindows()
        return

    print("未指定 --image/--dir/--cam")
    print("未指定 --image/--dir/--cam")
