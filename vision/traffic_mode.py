from __future__ import annotations  # noqa: I001

from pathlib import Path
from typing import Any, NamedTuple

import cv2

from detection.yolo_utils import YOLOAutoDetector, YOLOOpts, parse_img_size
from vision.color_detection import detect_traffic_light_color
from vision.roi_utils import (clamp_roi, iter_images_from_dir,
                              pick_roi_interactive)
from voice import Announcer

ROI = tuple[int, int, int, int]


def _color_bgr(name: str) -> tuple[int, int, int]:
    """将颜色名映射为 BGR 颜色元组（OpenCV 颜色顺序）。"""
    return (0, 200, 255) if name == "yellow" else ((0, 255, 0) if name == "green" else (0, 0, 255))


_ANN = Announcer(min_interval_sec=1.5)


def draw_and_report(win: str, frame, roi_rect: ROI) -> str:
    """在图像上绘制 ROI 与颜色文本，并返回判定的颜色字符串。"""
    x, y, w, h = roi_rect
    x, y, w, h = clamp_roi(x, y, w, h, (frame.shape[1], frame.shape[0]))
    roi = frame[y : y + h, x : x + w]
    color = detect_traffic_light_color(roi)
    out = frame.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(out, f"颜色：{color}", (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, _color_bgr(color), 2, cv2.LINE_AA)
    cv2.imshow(win, out)
    return color


def _save_crop(save_dir: Path, filename: str, roi) -> None:
    """保存 ROI 小图到指定目录。"""
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / filename), roi)
    except (OSError, cv2.error) as e:
        print(f"保存裁剪失败: {e}")


def _wait_key(delay: int = 0) -> int:
    """封装 cv2.waitKey 并屏蔽高位。"""
    return cv2.waitKey(delay) & 0xFF


def _put_status(img, text: str) -> None:
    """在左上角绘制状态文本。"""
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)


def _log_and_say(base_name: str, status: str, *, extra: str = "") -> None:
    """打印并通过 Announcer 播报状态。"""
    print(f"{base_name}：{status}{extra}")
    _ANN.say_traffic(status)


def _maybe_save_crop(args, *, is_cam: bool, base_name: str, box, img) -> None:
    """在传入了 --save-crops 时保存裁剪图。"""
    if not getattr(args, "save_crops", ""):
        return
    sd = Path(args.save_crops)
    x1, y1, x2, y2, _ = box
    _save_crop(sd, f"{('cam' if is_cam else base_name)}_{x1}_{y1}_{x2}_{y2}_{_box_color(img, box)}.jpg", img[y1:y2, x1:x2])


def _best_by_center(img, boxes):
    """按中心距离最小、置信度次关键倒序选择一个候选框。"""
    if not boxes:
        return None
    h_img, w_img = img.shape[:2]
    cx_img, cy_img = w_img / 2.0, h_img / 2.0

    def center_dist2(b):
        x1, y1, x2, y2, _ = b
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx, dy = cx - cx_img, cy - cy_img
        return dx * dx + dy * dy

    return min(boxes, key=lambda b: (center_dist2(b), -b[4]))


def _box_color(img, b) -> str:
    """对候选框区域进行颜色检测并返回类别名。"""
    x1, y1, x2, y2, _ = b
    roi = img[y1:y2, x1:x2]
    return detect_traffic_light_color(roi)


def _draw_box(out_img, b, label: str, color_name: str | None = None) -> None:
    """绘制候选框与标签文本。"""
    x1, y1, x2, y2, _ = b
    cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    color_disp = _color_bgr(color_name or "yellow")
    cv2.putText(out_img, label, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_disp, 2, cv2.LINE_AA)


def _horizontal_logic(img, out_img, hori_boxes):
    """处理水平样式交通灯的判定与绘制。"""
    if not hori_boxes:
        st = "无红绿灯"
        cv2.putText(out_img, st, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        return out_img, st

    if len(hori_boxes) > 1:
        colors = [_box_color(img, b) for b in hori_boxes]
        valid = [c for c in colors if c in {"red", "yellow", "green"}]
        if not valid:
            st = "红绿灯不工作"
            for b in hori_boxes:
                _draw_box(out_img, b, st)
            cv2.putText(out_img, st, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            return out_img, st
        uniq = set(valid)
        if len(uniq) == 1:
            st = next(iter(uniq))
            for b in hori_boxes:
                _draw_box(out_img, b, st, st)
            return out_img, st
        st = "颜色不同"
        for b, c in zip(hori_boxes, colors, strict=True):
            _draw_box(out_img, b, c if c in {"red", "yellow", "green"} else st, c if c in {"red", "yellow", "green"} else None)
        cv2.putText(out_img, st, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        return out_img, st

    hb = hori_boxes[0]
    c = _box_color(img, hb)
    st = c if c in {"red", "yellow", "green"} else "红绿灯不工作"
    _draw_box(out_img, hb, st, c if c in {"red", "yellow", "green"} else None)
    return out_img, st


class Scene(NamedTuple):
    img: Any
    out_img: Any
    vert_boxes: list[tuple[int, int, int, int, float]]
    hori_boxes: list[tuple[int, int, int, int, float]]


def _vertical_logic(scene: Scene, args, *, base_name: str, is_cam: bool):
    """处理竖直样式交通灯的判定与绘制。"""
    img, out_img, vert_boxes, hori_boxes = scene
    handled = False
    status: str | None = None
    if not vert_boxes:
        return out_img, status, handled
    handled = True
    if len(vert_boxes) > 1:
        vb = _best_by_center(img, vert_boxes)
        if vb is None:
            status = "无红绿灯"
            _put_status(out_img, status)
        else:
            c = _box_color(img, vb)
            if c in {"red", "yellow", "green"}:
                status = c
                _draw_box(out_img, vb, c, c)
                _maybe_save_crop(args, is_cam=is_cam, base_name=base_name, box=vb, img=img)
            else:
                out_img, status = _horizontal_logic(img, out_img, hori_boxes)
        return out_img, status, handled
    vb = vert_boxes[0]
    c = _box_color(img, vb)
    if c in {"red", "yellow", "green"}:
        status = c
        _draw_box(out_img, vb, c, c)
        _maybe_save_crop(args, is_cam=is_cam, base_name=base_name, box=vb, img=img)
        return out_img, status, handled
    if hori_boxes:
        out_img, status = _horizontal_logic(img, out_img, hori_boxes)
        return out_img, status, handled
    status = "红绿灯不工作"
    _draw_box(out_img, vb, status)
    return out_img, status, handled


def _auto_draw_and_save(
    detector: YOLOAutoDetector,
    img,
    args,
    base_name: str,
    *,
    is_cam: bool = False,
) -> tuple[Any, str]:
    """自动检测+绘制并按需保存裁剪图，返回输出图像与状态文本。"""
    vert_boxes, hori_boxes = detector.detect_orientations(img)
    out = img.copy()
    status: str | None = None
    out, status, handled = _vertical_logic(Scene(img, out, vert_boxes, hori_boxes), args, base_name=base_name, is_cam=is_cam)
    if not handled:
        if hori_boxes:
            out, status = _horizontal_logic(img, out, hori_boxes)
        else:
            status = "无红绿灯"
            _put_status(out, status)
    extra = ""
    if not (vert_boxes or hori_boxes):
        extra = f" (class-id={getattr(args, 'class_id', 9)})"
    _log_and_say(base_name, status or "无红绿灯", extra=extra)
    return out, (status or "无红绿灯")


def _handle_image(args, detector: YOLOAutoDetector | None, roi_rect: ROI | None, win: str) -> None:
    """处理单张图片模式。"""
    p = Path(args.image)
    if not p.is_file():
        raise FileNotFoundError(p)
    img = cv2.imread(str(p))
    if img is None:
        msg = f"无法读取图片: {p}"
        raise RuntimeError(msg)
    if detector is not None:
        out, _status = _auto_draw_and_save(detector, img, args, p.name)
        cv2.imshow(win, out)
        _wait_key(0)
    else:
        local_roi = roi_rect or pick_roi_interactive(win, img)
        if local_roi is None:
            print("未选择 ROI，退出。")
            cv2.destroyAllWindows()
            return
        color = draw_and_report(win, img, local_roi)
        print(f"{p}：{color}")
        _wait_key(0)
    cv2.destroyAllWindows()


def _handle_dir(args, detector: YOLOAutoDetector | None, roi_rect: ROI | None, win: str) -> None:
    """处理目录批量图片模式（支持 r 重选 ROI，q/Esc 退出）。"""
    dp = Path(args.dir)
    if not dp.is_dir():
        raise NotADirectoryError(dp)
    images = iter_images_from_dir(str(dp))
    if not images:
        print("目录中未找到图片 支持：.jpg .jpeg .png .bmp .webp")
        return
    idx = 0
    while idx < len(images):
        p = Path(images[idx])
        img = cv2.imread(str(p))
        if img is None:
            print(f"读取失败 跳过：{p}")
            idx += 1
            continue
        if detector is not None:
            out, _status = _auto_draw_and_save(detector, img, args, p.name)
            cv2.imshow(win, out)
            k = _wait_key(0)
        else:
            local_roi = roi_rect or pick_roi_interactive(win, img)
            if local_roi is None:
                print("未选择 ROI，退出。")
                break
            color = draw_and_report(win, img, local_roi)
            print(f"{p.name}：{color}")
            k = _wait_key(0)
        if k in {ord('q'), 27}:
            break
        if k == ord('r'):
            roi_rect = None
            continue
        idx += 1
    cv2.destroyAllWindows()


def _handle_cam(args, detector: YOLOAutoDetector | None, roi_rect: ROI | None, win: str) -> None:
    """处理摄像头实时模式（支持 r 重选 ROI，q/Esc 退出）。"""
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
                out, _status = _auto_draw_and_save(detector, frame, args, 'cam', is_cam=True)
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
    """交通灯颜色检测主流程入口（解析参数对象并分派对应模式）。"""
    win = getattr(args, 'win', 'traffic-test')
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    detector: YOLOAutoDetector | None = None
    if getattr(args, 'auto', False):
        opts = YOLOOpts(
            model_path=getattr(args, 'model', 'models/yolo/yolo11n.pt'),
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
