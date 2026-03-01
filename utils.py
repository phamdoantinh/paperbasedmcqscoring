"""
utils.py — Utility functions for the Paper-Based MCQ Scoring System (main branch)
==================================================================================
Merged from the original tool_algorithm.py and common_main.py files.

Contents
────────
  Constants         : display colours, confidence threshold
  Geometry          : order_points, find_dest, generate_output, custom_padding
  Class mapping     : get_class_marker, get_class_answer, get_class_info
  Duplicate removal : remove_elements_info/answer/marker
  Bounding-box      : get_coordinates, get_coordinates_info
  Question count    : get_parameter_number_anwser, get_remainder
  Orientation       : calculate_new_coordinates, orient_image_by_angle,
                      orient_image_step_by_step, rotate_image_by_angle
  Image helpers     : mergeImages, crop_image_answer, crop_image_info
"""

import os
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

warning_color     = (78, 173, 240)   # orange-ish (BGR)
blue_color        = (255, 0, 0)
red_color         = (0, 0, 255)
green_color       = (0, 255, 0)
threshold_warning = 0.79             # confidence below this → flag as uncertain


# ─────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────

def order_points(pts):
    """Order four corner points as [TL, TR, BR, BL]."""
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype("int").tolist()


def find_dest(pts):
    """Compute destination rectangle for perspective transform."""
    (tl, tr, br, bl) = pts
    widthA  = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB  = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth  = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    return order_points([[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]])


def custom_padding(corners, x):
    """Push each corner outward: TL→(-x,-x), TR→(+x,-x), BR→(+x,+x), BL→(-x,+x)."""
    return [
        [corners[0][0] - x, corners[0][1] - x],
        [corners[1][0] + x, corners[1][1] - x],
        [corners[2][0] + x, corners[2][1] + x],
        [corners[3][0] - x, corners[3][1] + x],
    ]


def generate_output(image: np.ndarray, corners: list) -> np.ndarray:
    """Perspective-crop the document region defined by four corners."""
    corners = order_points(corners)
    corners = custom_padding(corners, 40)
    destination_corners = find_dest(corners)
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    out = cv2.warpPerspective(
        image, M,
        (destination_corners[2][0], destination_corners[2][1]),
        flags=cv2.INTER_LANCZOS4,
    )
    return np.clip(out, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────
# Class label mapping  (YOLO class index → string label)
# ─────────────────────────────────────────────────────────────────

def get_class_marker(argument):
    """Map marker class index (0=marker1, 1=marker2) to its label string."""
    return {0: "marker1", 1: "marker2"}.get(argument, "")


def get_class_answer(argument):
    """
    Map answer model class index to its label string.

    0  → "x"  (unchoice / no selection)
    1–4  → A, B, C, D
    5–10 → AB, AC, AD, BC, BD, CD
    11–15 → ABC, ABD, ACD, BCD, ABCD
    """
    _MAP = {
        0:  "",
        1:  "A",   2:  "B",   3:  "C",   4:  "D",
        5:  "AB",  6:  "AC",  7:  "AD",  8:  "BC",  9:  "BD",  10: "CD",
        11: "ABC", 12: "ABD", 13: "ACD", 14: "BCD", 15: "ABCD",
    }
    return _MAP.get(argument, "")


def get_class_info(argument):
    """
    Map info model class index to its label string.

    Supports both compact (0–10) and extended (16–26) index ranges.
    0–9 / 16–25 → digit strings "0"–"9"
    10 / 26     → "x" (blank cell)
    """
    _MAP = {
        0: "0",  1: "1",  2: "2",  3: "3",  4: "4",
        5: "5",  6: "6",  7: "7",  8: "8",  9: "9",
        10: "x",
    }
    return _MAP.get(argument, "x")


# ─────────────────────────────────────────────────────────────────
# Duplicate-detection removal helpers
# ─────────────────────────────────────────────────────────────────

def remove_elements_info(arr):
    """Remove duplicate info-zone detections (close in x-axis)."""
    result, i = [], 0
    while i < len(arr):
        item = arr[i]
        result.append(item)
        j = i + 1
        while j < len(arr) and abs(item[0] - arr[j][0]) <= 5:
            if arr[j][4] >= item[4]:
                result.pop()
                break
            j += 1
        i = j
    return result


def remove_elements_answer(arr):
    """Remove duplicate answer detections (close in y-axis)."""
    result, i = [], 0
    while i < len(arr):
        item = arr[i]
        result.append(item)
        j = i + 1
        while j < len(arr) and abs(item[1] - arr[j][1]) <= 5:
            if arr[j][4] >= item[4]:
                result.pop()
                break
            j += 1
        i = j
    return result


def remove_elements_marker(arr):
    """Remove duplicate marker detections (close in both x and y)."""
    result, i = [], 0
    while i < len(arr):
        item = arr[i]
        result.append(item)
        j = i + 1
        while j < len(arr) and abs(item[0] - arr[j][0]) <= 5 and abs(item[1] - arr[j][1]) <= 5:
            if arr[j][4] >= item[4]:
                result.pop()
                break
            j += 1
        i = j
    return result


# ─────────────────────────────────────────────────────────────────
# Bounding-box coordinate helpers
# ─────────────────────────────────────────────────────────────────

def get_coordinates(x1, y1, x2, y2, class1):
    """Return bounding-box corners for each answer letter column."""
    offsets = {
        "A": (-5,  0, -15, 0),
        "B": (37,  0,  25, 0),
        "C": (75,  0,  68, 0),
        "D": (118, 0, 108, 0),
    }
    if class1 in offsets:
        dx1, _, dx2, _ = offsets[class1]
        w = (x2 - x1) // 4
        h = y2 - y1
        return x1 + dx1, y1 - 2, x1 + w + dx2, y1 + h
    return x1, y1, x2, y2


def get_coordinates_info(x1, y1, x2, y2, class1):
    """Return bounding-box corners for each digit row in the info zone."""
    digit_offsets = {str(d): d * 38 for d in range(10)}
    row_h = (y2 - y1) // 9
    if class1 in digit_offsets:
        dy = digit_offsets[class1]
        return x1, y1 + dy, x2, y1 + row_h + dy
    return x1, y1, x2, y2


# ─────────────────────────────────────────────────────────────────
# Question-count helpers
# ─────────────────────────────────────────────────────────────────

def get_parameter_number_anwser(numberAnswer):
    """Return the number of answer-column images needed (floor(n/20))."""
    return numberAnswer // 20


def get_remainder(numberAnswer):
    """Return the number of questions in the partial last column."""
    return numberAnswer % 20


# ─────────────────────────────────────────────────────────────────
# Orientation helpers
# ─────────────────────────────────────────────────────────────────

def calculate_new_coordinates(marker_coordinates, rect, param1, param2):
    """Compute the padded centre of the bounding box matching `rect`."""
    matching_indices = np.where((marker_coordinates[:, :2] == rect).all(axis=1))
    c = marker_coordinates[matching_indices].flatten()
    return np.array([(c[0] + c[2]) / 2 + param1, (c[1] + c[3]) / 2 + param2])


def orient_image_by_angle(pts, marker_coordinates):
    """Legacy orientation helper — orders the four marker points as [TL,TR,BR,BL]."""
    rect = np.zeros((4, 2), dtype="float32")
    marker_coordinates_true = []
    param = 40
    pts = np.array(pts)
    marker_coordinates = np.array(marker_coordinates)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    marker_coordinates_true.append(calculate_new_coordinates(marker_coordinates, rect[0], -param, -param))
    rect[2] = pts[np.argmax(s)]
    marker_coordinates_true.append(calculate_new_coordinates(marker_coordinates, rect[2], param, param))
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    marker_coordinates_true.append(calculate_new_coordinates(marker_coordinates, rect[1], param, -param))
    rect[3] = pts[np.argmax(diff)]
    marker_coordinates_true.append(calculate_new_coordinates(marker_coordinates, rect[3], -param, param))
    marker_coordinates_true = np.array([marker_coordinates_true]).reshape(-1, 1, 2)
    return rect.astype("int").tolist(), marker_coordinates_true


def orient_image_step_by_step(pts, marker_coordinates, marker2_position):
    """
    Determine the correct rotation angle to straighten the answer sheet.

    Steps
    ─────
    1. P3 = marker2 (Bottom-Right reference corner).
    2. Distances from P3 to each of the 3 marker1 points.
    3. P4 (Bottom-Left) = closest marker1 point to P3.
    4. P4ʹ = (xP3 − d1, yP3)  — ideal horizontal position of P4.
    5. dP4P4ʹ.
    6. α = arccos((2·d1² − dP4P4ʹ²) / 2·d1²); negate if P4 is below P4ʹ.
    7. Return padded marker coordinates and α for warpAffine.
    """
    pts               = np.array(pts, dtype="float32")
    marker_coordinates = np.array(marker_coordinates)
    marker2_position  = np.array(marker2_position, dtype="float32")

    P3 = marker2_position
    xP3, yP3 = P3[0], P3[1]

    marker1_points = np.array(
        [pt for pt in pts
         if not (abs(pt[0] - marker2_position[0]) < 1e-6
                 and abs(pt[1] - marker2_position[1]) < 1e-6)],
        dtype="float32",
    )

    distances = [np.linalg.norm(P3 - pt) for pt in marker1_points]
    d1 = min(distances)

    P4_idx = int(np.argmin(distances))
    P4     = marker1_points[P4_idx]

    remaining = [pt for i, pt in enumerate(marker1_points) if i != P4_idx]
    remaining.sort(key=lambda p: p[0])
    P1, P2 = remaining[0], remaining[1]

    P4_prime    = np.array([xP3 - d1, yP3], dtype="float32")
    dP4P4_prime = np.linalg.norm(P4 - P4_prime)

    denominator = 2 * d1 ** 2
    if denominator != 0:
        cos_alpha     = np.clip((2 * d1 ** 2 - dP4P4_prime ** 2) / denominator, -1.0, 1.0)
        alpha_degrees = float(np.degrees(np.arccos(cos_alpha)))
        if P4[1] > P4_prime[1]:
            alpha_degrees = -alpha_degrees
    else:
        alpha_degrees = 0.0

    rect = [P1, P2, P3, P4]
    param = 0
    marker_coordinates_true = [
        calculate_new_coordinates(marker_coordinates, rect[0], -param, -param),
        calculate_new_coordinates(marker_coordinates, rect[1],  param, -param),
        calculate_new_coordinates(marker_coordinates, rect[2],  param,  param),
        calculate_new_coordinates(marker_coordinates, rect[3], -param,  param),
    ]
    marker_coordinates_true = np.array([marker_coordinates_true]).reshape(-1, 1, 2)
    return marker_coordinates_true, alpha_degrees


def rotate_image_by_angle(image, angle_degrees, center=None):
    """
    Rotate `image` by `angle_degrees` without clipping any corner.

    Args:
        image         : Input BGR numpy array.
        angle_degrees : Positive = counter-clockwise, negative = clockwise.
        center        : Centre of rotation (x, y). Defaults to image centre.

    Returns:
        (rotated_image, rotation_matrix)
    """
    if image is None:
        raise ValueError("Input image must not be None")
    height, width = image.shape[:2]
    if center is None:
        center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    new_width  = int(height * sin_val + width  * cos_val)
    new_height = int(height * cos_val + width  * sin_val)
    rotation_matrix[0, 2] += new_width  / 2 - center[0]
    rotation_matrix[1, 2] += new_height / 2 - center[1]
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (new_width, new_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rotated_image, rotation_matrix


# ─────────────────────────────────────────────────────────────────
# Image crop / merge helpers  (formerly common_main.py)
# ─────────────────────────────────────────────────────────────────

def mergeImages(filename, coord_array, array_img_graft, background_image, imgInfo, args):
    """
    Overlay annotated info-zone and answer-column crops back onto the
    full document image and save the result to HandledSheets/.
    """
    file_extension = filename.split(".")[-1]
    filename_cut   = filename.rsplit(".", 1)[0]
    h, w, _        = imgInfo.shape
    background_image[0:h, 500:500 + w] = imgInfo / 255
    for i, (x_coord, y_coord) in enumerate(coord_array):
        h, w, _ = array_img_graft[i].shape
        background_image[y_coord:y_coord + h, x_coord:x_coord + w] = array_img_graft[i] / 255
    handled_path = (
        f"images/answer_sheets/{args.input}/HandledSheets/"
        f"handled_{filename_cut}.{file_extension}"
    )
    cv2.imwrite(handled_path, background_image * 255)
    return handled_path


def crop_image_answer(img, numberAnswer):
    """
    Crop the three answer-column regions from a 1056×1500 document image,
    resize each to 250×640 for model inference.
    """
    arrayX = [30, 350, 660]
    ans_blocks = [
        img[480: 480 + 896, arrayX[i]: arrayX[i] + 350]
        for i in range(3)
    ]
    n_cols = get_parameter_number_anwser(numberAnswer)
    if numberAnswer not in (20, 40, 60):
        n_cols += 1
    ans_blocks = ans_blocks[:n_cols]

    resized, size_array, coord_array = [], [], []
    for i, block in enumerate(ans_blocks):
        size_array.append((350, 896))
        resized.append(cv2.resize(block, (250, 640), interpolation=cv2.INTER_AREA))
        coord_array.append((arrayX[i], 480))
    return resized, size_array, coord_array


def crop_image_info(img):
    """
    Crop the info zone (class code / student ID / exam code) from a
    normalised document image (pixel values in [0, 1]).
    Converts back to uint8 and resizes to 640×640.
    """
    cropped = img[0:500, 500:1006]
    cropped = cv2.convertScaleAbs(cropped * 255)
    return cv2.resize(cropped, (640, 640), interpolation=cv2.INTER_AREA)
