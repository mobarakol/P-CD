import cv2
import numpy as np


def max_contour(img):
    norm_image = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)

    binary = cv2.threshold(norm_image, 127, 255, cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    output = np.zeros_like(img)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(output, [c], -1, 255, -1)

    return output
