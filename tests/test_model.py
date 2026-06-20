import sys
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from ultralytics import settings
settings.update({'sync': False})
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
from perception import DualModelPerception
def main():
    model_a_path = PROJECT_ROOT / "models" / "v11-optimized-fast" / "weights" / "best.pt"
    if not model_a_path.exists():
        model_a_path = PROJECT_ROOT / "models" / "gerobak_best.pt"
    model_b_path = PROJECT_ROOT / "models" / "yolov11x-seg.pt"
    if not model_b_path.exists():
        model_b_path = PROJECT_ROOT / "models" / "yolo11n-seg.pt"
    image_path = './data/test-img2.jpg'
    print("Initializing PyTorch...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"\nLoading dual models...")
    print(f"Model A (Gerobak): {model_a_path}")
    print(f"Model B (Environment): {model_b_path}")
    start_time = time.time()
    perception = DualModelPerception(
        gerobak_weights=model_a_path,
        env_weights=model_b_path,
        gerobak_conf=0.45,
        env_conf=0.45,
        device=device
    )
    end_time = time.time()
    print(f"Models loaded successfully in {end_time - start_time:.2f} seconds!")
    print("\nRunning inference...")
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image from {image_path}")
        output = perception.predict(frame)
        res_plotted = frame.copy()
        for det in output.sidewalk:
            mask = det.mask
            res_plotted[mask > 0] = (0, 220, 0)
        for det in output.gerobak:
            if det.class_name.lower().strip() in ["sidewalk", "trotoar", "pavement"]:
                res_plotted[det.mask > 0] = (0, 220, 0)
            else:
                res_plotted[det.mask > 0] = (0, 0, 220)
        mask_alpha = 0.45
        res_plotted = cv2.addWeighted(res_plotted, mask_alpha, frame, 1.0 - mask_alpha, 0)
        for det in output.gerobak:
            is_sidewalk = det.class_name.lower().strip() in ["sidewalk", "trotoar", "pavement"]
            color = (0, 255, 0) if is_sidewalk else (0, 0, 255)
            contours, _ = cv2.findContours(det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(res_plotted, contours, -1, color, 2)
            if len(contours) > 0:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    label = f"{det.class_name} {det.score:.2f}"
                    cv2.putText(res_plotted, label, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        for det in output.sidewalk:
            contours, _ = cv2.findContours(det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(res_plotted, contours, -1, (0, 255, 0), 2)
        print("Inference successful. Saving results to 'inference_result.jpg'...")
        cv2.imwrite("inference_result.jpg", res_plotted)
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Please ensure the image_path is correct and points to a valid image.")
if __name__ == "__main__":
    main()
