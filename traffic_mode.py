"""交通灯颜色测试模式的实现。

提供 run(args) 给 test.py 调用，支持：
- 图片/目录/摄像头
- 手动 ROI（交互/参数）或 YOLO 自动识别裁剪
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from color_detction import detect_traffic_light_color
from roi_utils import clamp_roi, iter_images_from_dir, pick_roi_interactive
from yolo_utils import YOLOAutoDetector, YOLOOpts, parse_img_size

# 类型别名
ROI = tuple[int, int, int, int]


def _color_bgr(name: str) -> tuple[int, int, int]:
    return (0, 200, 255) if name == "yellow" else ((0, 255, 0) if name == "green" else (0, 0, 255))


def draw_and_report(win: str, frame, roi_rect: ROI) -> str:
    x, y, w, h = roi_rect
    x, y, w, h = clamp_roi(x, y, w, h, (frame.shape[1], frame.shape[0]))
    roi = frame[y : y + h, x : x + w]
    color = detect_traffic_light_color(roi)
    out = frame.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(out, f"color: {color}", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, _color_bgr(color), 2, cv2.LINE_AA)
    cv2.imshow(win, out)
    return color


def _save_crop(save_dir: Path, filename: str, roi) -> None:
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / filename), roi)
    except (OSError, cv2.error) as e:  # type: ignore[attr-defined]
        print(f"保存裁剪失败: {e}")


def _wait_key(delay: int = 0) -> int:
    """显示后等待按键，返回按键码（低 8 位）。"""
    return cv2.waitKey(delay) & 0xFF


def _auto_draw_and_save(
    detector: YOLOAutoDetector,
    img,
    args,
    base_name: str,
    *,
    is_cam: bool = False,
) -> tuple[Any, bool]:
    """运行 YOLO，绘制框体、保存裁剪，并打印结果。

    返回 (out_image, has_boxes)。
    """
    boxes = detector.detect(img)
    out = img.copy()
    has_boxes = bool(boxes)
    if not has_boxes:
        if getattr(detector, 'last_state', '') == 'no_vertical':
            print(f"{base_name}: 检测到红绿灯但无竖直方向目标")
        else:
            print(f"{base_name}: 未检测到交通灯 (class-id={getattr(args, 'class_id', 9)})")
        return out, False

    for (x1, y1, x2, y2, conf) in boxes:
        roi = img[y1:y2, x1:x2]
        color = detect_traffic_light_color(roi)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(out, f"{color} {conf:.2f}", (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _color_bgr(color), 2, cv2.LINE_AA)
        if getattr(args, 'save_crops', ''):
            sd = Path(args.save_crops)
            filename = (
                f"cam_{x1}_{y1}_{x2}_{y2}_{color}.jpg" if is_cam else f"{base_name}_{x1}_{y1}_{x2}_{y2}_{color}.jpg"
            )
            _save_crop(sd, filename, roi)
        print(f"{base_name}: {color} (conf={conf:.2f}) box=({x1}, {y1}, {x2}, {y2})")
    return out, True


def _handle_image(args, detector: YOLOAutoDetector | None, roi_rect: ROI | None, win: str) -> None:
    p = Path(args.image)
    if not p.is_file():
        raise FileNotFoundError(p)
    img = cv2.imread(str(p))
    if img is None:
        msg = f"无法读取图片: {p}"
        raise RuntimeError(msg)
    if detector is not None:
        out, _ = _auto_draw_and_save(detector, img, args, p.name)
        cv2.imshow(win, out)
        _wait_key(0)
    else:
        local_roi = roi_rect or pick_roi_interactive(win, img)
        if local_roi is None:
            print("未选择 ROI，退出。")
            cv2.destroyAllWindows()
            return
        color = draw_and_report(win, img, local_roi)
        print(f"{p}: {color}")
        _wait_key(0)
    cv2.destroyAllWindows()


def _handle_dir(args, detector: YOLOAutoDetector | None, roi_rect: ROI | None, win: str) -> None:
    dp = Path(args.dir)
    if not dp.is_dir():
        raise NotADirectoryError(dp)
    images = iter_images_from_dir(str(dp))
    if not images:
        print("目录中未找到图片。支持: .jpg .jpeg .png .bmp .webp")
        return
    idx = 0
    while idx < len(images):
        p = Path(images[idx])
        img = cv2.imread(str(p))
        if img is None:
            print(f"读取失败，跳过: {p}")
            idx += 1
            continue
        if detector is not None:
            out, _ = _auto_draw_and_save(detector, img, args, p.name)
            cv2.imshow(win, out)
            k = _wait_key(0)
        else:
            local_roi = roi_rect or pick_roi_interactive(win, img)
            if local_roi is None:
                print("未选择 ROI，退出。")
                break
            color = draw_and_report(win, img, local_roi)
            print(f"{p.name}: {color}")
            k = _wait_key(0)
        if k in {ord('q'), 27}:
            break
        if k == ord('r'):
            roi_rect = None
            continue
        idx += 1
    cv2.destroyAllWindows()


def _handle_cam(args, detector: YOLOAutoDetector | None, roi_rect: ROI | None, win: str) -> None:
    cap = cv2.VideoCapture(int(args.cam))
    if not cap.isOpened():
        msg = f"无法打开摄像头 {args.cam}"
        raise RuntimeError(msg)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("读取摄像头帧失败")
                break
            if detector is not None:
                out, has_boxes = _auto_draw_and_save(detector, frame, args, 'cam', is_cam=True)
                if not has_boxes:
                    info = '无竖向红绿灯' if getattr(detector, 'last_state', '') == 'no_vertical' else '未检测到红绿灯'
                    cv2.putText(out, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(win, out)
            else:
                local_roi = roi_rect or pick_roi_interactive(win, frame)
                if local_roi is None:
                    print("未选择 ROI，退出。")
                    break
                roi_rect = local_roi
                draw_and_report(win, frame, roi_rect)
            k = _wait_key(max(1, int(getattr(args, 'delay', 1))))
            if k in {ord('q'), 27}:
                break
            if k == ord('r'):
                roi_rect = None
    finally:
        cap.release()
    cv2.destroyAllWindows()


def run(args) -> None:
    """入口：根据参数选择图片/目录/摄像头模式，并委托给对应处理器。"""
    win = getattr(args, 'win', 'traffic-test')
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # YOLO 自动检测（按需）
    detector: YOLOAutoDetector | None = None
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

    roi_rect: ROI | None = tuple(args.roi) if getattr(args, 'roi', None) else None

    if getattr(args, 'image', None):
        _handle_image(args, detector, roi_rect, win)
        return

    if getattr(args, 'dir', None):
        _handle_dir(args, detector, roi_rect, win)
        return

    if getattr(args, 'cam', None) is not None:
        _handle_cam(args, detector, roi_rect, win)
        return

    print("未指定 --image/--dir/--cam")
