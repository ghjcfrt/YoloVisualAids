"""交通灯颜色测试模式的实现。

提供 run(args) 给 test.py 调用，支持：
- 图片/目录/摄像头
- 手动 ROI（交互/参数）或 YOLO 自动识别裁剪
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import cv2

from announce import Announcer
from color_detction import detect_traffic_light_color
from roi_utils import clamp_roi, iter_images_from_dir, pick_roi_interactive
from yolo_utils import YOLOAutoDetector, YOLOOpts, parse_img_size

# 类型别名
ROI = tuple[int, int, int, int]


def _color_bgr(name: str) -> tuple[int, int, int]:
    return (0, 200, 255) if name == "yellow" else ((0, 255, 0) if name == "green" else (0, 0, 255))


# 语音播报器（模块内复用，避免重复初始化）
_ANN = Announcer(min_interval_sec=1.5)


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


def _put_status(img, text: str) -> None:
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)


def _log_and_say(base_name: str, status: str, *, extra: str = "") -> None:
    print(f"{base_name}: {status}{extra}")
    _ANN.say_traffic(status)


def _maybe_save_crop(args, *, is_cam: bool, base_name: str, box, img) -> None:
    if not getattr(args, 'save_crops', ''):
        return
    sd = Path(args.save_crops)
    x1, y1, x2, y2, _ = box
    _save_crop(sd, f"{('cam' if is_cam else base_name)}_{x1}_{y1}_{x2}_{y2}_{_box_color(img, box)}.jpg", img[y1:y2, x1:x2])


def _best_by_center(img, boxes):
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
    x1, y1, x2, y2, _ = b
    roi = img[y1:y2, x1:x2]
    return detect_traffic_light_color(roi)


def _draw_box(out_img, b, label: str, color_name: str | None = None) -> None:
    x1, y1, x2, y2, _ = b
    cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    color_disp = _color_bgr(color_name or 'yellow')
    cv2.putText(out_img, label, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_disp, 2, cv2.LINE_AA)


def _horizontal_logic(img, out_img, hori_boxes):
    """基于横向灯集合决策并绘制，返回 (out_img, status)。不负责播报/打印。"""
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

    # 单灯情形
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
    """基于竖向灯集合决策并绘制，返回 (out_img, status, handled)。"""
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
    # 恰好 1 个竖向
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
    """运行 YOLO 并按指定规则决策，绘制、可选保存裁剪，并返回状态信号。

    状态信号只会是以下之一：
    - 'red' | 'yellow' | 'green'
    - '颜色不同' | '红绿灯不工作' | '无红绿灯'
    """
    vert_boxes, hori_boxes = detector.detect_orientations(img)
    out = img.copy()
    status: str | None = None

    # 1) 是否存在竖向红绿灯？
    out, status, handled = _vertical_logic(Scene(img, out, vert_boxes, hori_boxes), args, base_name=base_name, is_cam=is_cam)
    if not handled:
        if hori_boxes:
            out, status = _horizontal_logic(img, out, hori_boxes)
        else:
            status = "无红绿灯"
            _put_status(out, status)

    # 统一打印与播报
    extra = ""
    if not (vert_boxes or hori_boxes):
        extra = f" (class-id={getattr(args, 'class_id', 9)})"
    _log_and_say(base_name, status or "无红绿灯", extra=extra)
    return out, (status or "无红绿灯")


def _handle_image(args, detector: YOLOAutoDetector | None, roi_rect: ROI | None, win: str) -> None:
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
            out, _status = _auto_draw_and_save(detector, img, args, p.name)
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
