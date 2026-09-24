import cv2
import time
from picamera2 import Picamera2

def capture_image(frame):
    filename = f"capture_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Image saved as {filename}")

def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640,480), "format": "RGB888"},
        controls={"FrameRate": 30}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    while True:
        frame = picam2.capture_array()

        cv2.imshow("Video", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):   # press 1 to capture
            capture_image(frame)

        elif key == ord('q'): # press q to quit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
