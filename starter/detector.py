import cv2


class MotionDetector:
    """Very small baseline detector. Candidates are expected to improve/replace it."""

    def __init__(self, min_area=900):
        self.min_area = min_area
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=32, detectShadows=True
        )

    def detect(self, frame):
        mask = self.subtractor.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append({
                "bbox": [x, y, w, h],
                "centroid": [x + w // 2, y + h // 2],
                "confidence": 0.50,
                "object_type": "moving_object",
            })
        return detections
