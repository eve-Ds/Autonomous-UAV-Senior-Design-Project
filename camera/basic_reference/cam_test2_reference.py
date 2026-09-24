import asyncio
import cv2
from picamera2 import Picamera2            
from ultralytics import YOLO
from tracking import mult_detect_targets, debug_vis_multi, HybridTracker
import time


err_min = 20
flight_time = 100    # flight time



async def main():
    model = YOLO("best7.pt")

    await asyncio.sleep(2)

    # Picamera video feed
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888", },
        controls={"FrameRate": 30}
    )
    picam2.configure(config)
    picam2.start()
    await asyncio.sleep(2)

    ht = HybridTracker(detect_every=2, conf=0.30, classes=None,
                       max_missed=2, match_iou_thresh=0.02)

    start_time = time.time()

    while (time.time() - start_time < flight_time):
        frame = picam2.capture_array()
        targets, frame = mult_detect_targets(frame, model, ht)

        vis = debug_vis_multi(frame, targets)
        cv2.imshow("Multi-Detect", vis)
        

        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
