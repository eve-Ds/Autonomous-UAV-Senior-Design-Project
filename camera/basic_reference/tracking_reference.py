import cv2
import numpy as np


class HybridTracker:
    def __init__(self, detect_every=3, conf=0.2, classes=None, max_missed=2, match_iou_thresh=0.2,):
        self.tracker = None
        self.bbox = None  # (x, y, w, h)

        self.frames_since_detect = 0
        self.detect_every = detect_every
        self.conf = conf
        self.classes = classes

        self.missed_frames = 0
        self.target_lost = True
        self.max_missed = max_missed
        self.match_iou_thresh = match_iou_thresh

    def _create_tracker(self):
        legacy = getattr(cv2, "legacy", None)

        for mod in (legacy, cv2):
            if mod is None:
                continue
            for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMOSSE_create"):
                fn = getattr(mod, name, None)
                if fn is not None:
                    return fn()

        raise RuntimeError("No OpenCV tracker available (need opencv-contrib).")

    def _init_tracker(self, frame, bbox):
        self.tracker = self._create_tracker()
        self.tracker.init(frame, bbox)
        self.bbox = bbox
        self.frames_since_detect = 0

## Intersection over Union
def bbox_iou(a, b):
    # Compute IoU(Intersection over union) between two boxes in (x, y, w, h) format
    # IoU measures how much the boxes overlap:
    #   1.0 = perfect overlap
    #   0.0 = no overlap
    # This is used to check whether the tracker box and a detected box
    # refer to the same target

    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    union = (aw * ah) + (bw * bh) - inter
    return inter / union if union > 0 else 0.0


def best_matching_detection(tracker_bbox, detections, iou_thresh=0.2):
    # Compare the current tracker box against all detected boxes and
    # choose the detection with the highest IoU
    # If the best overlap is above the threshold, treat it as a valid match
    # Otherwise return no match
    
    best_bbox = None
    best_conf = 0.0
    best_iou = 0.0

    for det_bbox, conf in detections:
        iou = bbox_iou(tracker_bbox, det_bbox)
        if iou > best_iou:
            best_iou = iou
            best_bbox = det_bbox
            best_conf = conf

    if best_iou >= iou_thresh:
        return best_bbox, best_conf, best_iou

    return None, None, 0.0


def process_frame(frame, model, ht: HybridTracker):
    # Detects the target then gives the bounding box to the tracker
    # Returns the box offset from the center of frame

    if frame is None or frame.size == 0:
        return None, None, frame, "invalid"

    source = None
    tracker_bbox = None
    tracked_ok = False

    # 1) Tracker update
    if ht.tracker is not None:
        tracked_ok, bbox = ht.tracker.update(frame)
        if tracked_ok:
            tracker_bbox = tuple(map(float, bbox))
        else:
            ht.tracker = None
            ht.bbox = None

    # 2) Decide whether to detect
    detect_interval = 1 if ht.missed_frames > 0 else ht.detect_every
    need_detect = (ht.tracker is None) or (ht.frames_since_detect >= detect_interval)

    detections = []
    if need_detect:
        ht.frames_since_detect = 0

        res = model(frame, verbose=False, conf=ht.conf, classes=ht.classes)[0]

        for b in res.boxes:
            conf = float(b.conf.item())
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()

            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)

            if w > 0 and h > 0:
                detections.append(((x, y, w, h), conf))

            # Get class index (integer)
            class_id = int(b.cls[0])
            # Get class name using the index
            class_name = res.names[class_id]
            print(f"Detected: {class_name}")

        # 3) If tracker exists, try to confirm it with detection
        if tracker_bbox is not None and detections:
            match_bbox, _, _ = best_matching_detection(
                tracker_bbox,
                detections,
                iou_thresh=ht.match_iou_thresh,
            )

            if match_bbox is not None:
                ht._init_tracker(frame, match_bbox)
                ht.bbox = match_bbox
                ht.missed_frames = 0
                ht.target_lost = False
                source = "track+detect"
            else:
                ht.missed_frames += 1

                if ht.missed_frames >= ht.max_missed:
                    ht.tracker = None
                    ht.bbox = None
                    ht.target_lost = True
                    source = "lost"
                else:
                    ht.bbox = tracker_bbox
                    ht.target_lost = False
                    source = "uncertain"

        # 4) No tracker, but detector found something -> acquire target
        elif detections:
            best_det_bbox, _ = max(detections, key=lambda d: d[1])
            ht._init_tracker(frame, best_det_bbox)
            ht.bbox = best_det_bbox
            ht.missed_frames = 0
            ht.target_lost = False
            source = "detect"

        # 5) No tracker and no detection -> lost
        else:
            ht.missed_frames += 1
            ht.tracker = None
            ht.bbox = None
            ht.target_lost = True
            source = "lost"

    else:
        # No detection this frame, trust tracker for now
        ht.frames_since_detect += 1

        if tracker_bbox is not None:
            ht.bbox = tracker_bbox
            ht.target_lost = False
            source = "track"
        else:
            ht.missed_frames += 1
            ht.bbox = None
            ht.target_lost = True
            source = "lost"

    # 6) Output only valid target data
    bbox = None if ht.target_lost else ht.bbox

    err_px = None
    if bbox is not None:
        x, y, w, h = bbox
        H, W = frame.shape[:2]

        cx = x + w / 2.0
        cy = y + h / 2.0
        err_px = (cx - W / 2.0, cy - H / 2.0)

    return bbox, err_px, frame, source


def debug_vis(frame, bbox=None, err_px=None, source=None, missed_frames=0, target_lost=False):
    # Debug visual for testing

    vis = frame.copy()

    if bbox is not None:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if err_px is not None:
        H, W = vis.shape[:2]
        cx = int(W / 2 + err_px[0])
        cy = int(H / 2 + err_px[1])

        cv2.circle(vis, (W // 2, H // 2), 5, (255, 0, 0), -1)
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(vis, (W // 2, H // 2), (cx, cy), (255, 255, 0), 2)

    status = f"Source: {source}  Missed: {missed_frames}  Lost: {target_lost}"
    cv2.putText(
        vis,
        status,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return vis




def mult_detect_targets(frame, model, ht: HybridTracker):
    # Detects the target then gives the bounding box
    # Returns the box offset from the center of frame

    if frame is None or frame.size == 0:
        return None, frame

    res = model(frame, verbose=False, conf=ht.conf, classes=ht.classes)[0]

    H, W = frame.shape[:2]
    frame_cx = W / 2.0
    frame_cy = H / 2.0

    targets = []

    for b in res.boxes:

        conf = float(b.conf.item())

        if conf < ht.conf:
            continue
        
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()

        x = int(x1)
        y = int(y1)
        w = int(x2 - x1)
        h = int(y2 - y1)
        
        if w <= 0 or h <= 0:
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0
        err_px = (cx - frame_cx, cy - frame_cy)

        # Get class info
        class_id = int(b.cls[0])
        class_name = res.names[class_id]

        targets.append({
            "bbox": (x,y,w,h),
            "class": class_name,
            "error": err_px,
            "conf": conf
        })

    return targets, frame


def debug_vis_multi(frame, targets):
    vis = frame.copy()
    H, W = vis.shape[:2]

    # Draw center once
    cv2.circle(vis, (W // 2, H // 2), 5, (255, 0, 0), -1)

    for t in targets:
        x, y, w, h = map(int, t["bbox"])
        err_px = t["error"]
        cls = t["class"]

        # Box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Target center
        cx = int(W / 2 + err_px[0])
        cy = int(H / 2 + err_px[1])
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

        # Line to center
        cv2.line(vis, (W // 2, H // 2), (cx, cy), (255, 255, 0), 2)

        # Label
        cv2.putText(
            vis,
            f"{cls}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return vis


def crop_detect_targets(frame, identify, det_model, cls_model=None, conf=0.2, classes=None):

    if frame is None or frame.size == 0:
        return [], frame, "invalid"

    if not identify:
        return [], frame

    res = det_model(frame, verbose=False, conf=conf, classes=classes)[0]

    H, W = frame.shape[:2]
    frame_cx, frame_cy = W / 2.0, H / 2.0

    targets = []

    for b in res.boxes:
        conf_val = float(b.conf.item())
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()

        x, y = int(x1), int(y1)
        w, h = int(x2 - x1), int(y2 - y1)

        if w <= 0 or h <= 0:
            continue

        # Crop
        crop = frame[y:y+h, x:x+w]

        #  Default class (from detector)
        class_id = int(b.cls[0])
        class_name = res.names[class_id]

        refined_class = class_name
        refined_conf = conf_val

        #  Re-classify (if classifier provided)
        if cls_model is not None and crop.size > 0:
            cls_res = cls_model(crop)
            
            refined_class = cls_res[0].names[int(cls_res[0].probs.top1)]
            refined_conf = float(cls_res[0].probs.top1conf)

        # Localization
        cx = x + w / 2.0
        cy = y + h / 2.0

        err_px = (cx - frame_cx, cy - frame_cy)

        targets.append({
            "bbox": (x, y, w, h),
            "error": err_px,
            "class": refined_class,
            "conf": refined_conf
        })

    return targets, frame
