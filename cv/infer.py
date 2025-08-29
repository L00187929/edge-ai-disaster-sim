import cv2
import numpy as np

def analyse(arr):
    h, w = arr.shape[:2]
    total = float(h*w)
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
    hch, sch, vch = cv2.split(hsv)

    fire_mask = ((arr[:,:,2] > arr[:,:,1]) & (arr[:,:,2] > arr[:,:,0]) & (sch > 100))
    fire_ratio = fire_mask.sum() / total

    flood_mask = (arr[:,:,0] > arr[:,:,1]) & (arr[:,:,0] > arr[:,:,2])
    flood_ratio = flood_mask.sum() / total

    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = edges.sum() / (255.0*total)

    label = "unknown"
    score = 0.3

    if fire_ratio > 0.12:
        label, score = "fire", min(0.99, 0.5 + fire_ratio)
    if flood_ratio > 0.20:
        label, score = "flood", min(0.99, 0.5 + flood_ratio)
    if edge_ratio > 0.10:
        if label == "unknown":
            label, score = "damage", min(0.95, 0.4 + edge_ratio)
        else:
            score = min(0.99, score + 0.1)

    return {
        "label": label, "confidence": float(score),
        "metrics": {"fire_ratio": float(fire_ratio),
                    "flood_ratio": float(flood_ratio),
                    "edge_ratio": float(edge_ratio)}
    }
