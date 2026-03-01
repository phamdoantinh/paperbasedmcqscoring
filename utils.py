"""
utils.py — Utility functions for the Paper-Based MCQ Scoring System
====================================================================
Merged from the original tool_algorithm.py and common_main.py files.

Contents
────────
  Constants       : display colours, confidence threshold
  Geometry        : order_points, find_dest, generate_output, custom_padding
  Class mapping   : get_class, get_class_marker
  Duplicate removal : remove_elements_info/answer/marker
  Bounding-box    : get_coordinates, get_coordinates_info
  Question count  : get_parameter_number_anwser, get_remainder
  Orientation     : calculate_new_coordinates, orient_image_by_angle,
                    orient_image_step_by_step, rotate_image_by_angle
  Image helpers   : mergeImages, crop_image_answer, crop_image_info
"""

import os
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

warning_color    = (78, 173, 240)   # orange-ish (BGR)
blue_color       = (255, 0, 0)
red_color        = (0, 0, 255)
green_color      = (0, 255, 0)
threshold_warning = 0.79            # confidence below this → flag as uncertain


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
    """
    Apply directional padding to push each corner outward:
      TL → (-x, -x), TR → (+x, -x), BR → (+x, +x), BL → (-x, +x)
    """
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


def get_class(argument):
    """
    Map YOLO class index (0–28) to its label string.

    0        → "" (unchoice / no selection)
    1–15     → answer combinations: A, B, C, D, AB, AC, AD, BC, BD, CD,
                                     ABC, ABD, ACD, BCD, ABCD
    16–25    → info-zone digits: "0" … "9"
    26       → "x"  (blank info cell)
    27       → "marker1"
    28       → "marker2"
    """
    _MAP = {
        0:  "",
        1:  "A",   2:  "B",   3:  "C",   4:  "D",
        5:  "AB",  6:  "AC",  7:  "AD",  8:  "BC",  9:  "BD",  10: "CD",
        11: "ABC", 12: "ABD", 13: "ACD", 14: "BCD", 15: "ABCD",
        16: "0",   17: "1",   18: "2",   19: "3",   20: "4",
        21: "5",   22: "6",   23: "7",   24: "8",   25: "9",
        26: "x",
        27: "marker1", 28: "marker2",
    }
    return _MAP.get(argument, "")


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
# Bounding-box coordinate helpers (for annotated output images)
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
    """
    Legacy orientation helper (not used by the main scoring pipeline).
    Orders the four marker points as [TL, TR, BR, BL] and applies padding.
    """
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
    New function that performs each step as required:
    Step 1: Get the coordinates of marker2 (P3) and treat it as the new BR
    Step 2: Find d1, the minimum distance from P3 (marker2) to the 3 remaining points
    Step 3: Derive the coordinates of P4 (bottom-left)
    Step 4: Determine the coordinates of P4', where xP4' = xP3 - d1, yP4' = yP3
    Step 5: Calculate the distance dP4P4' from P4 to P4'
    Step 6: Calculate angle alpha = cos^-1((2*d1^2 - dP4P4'^2) / (2*d1^2))
    Step 7: Rotate the input image by angle alpha
    """
    
    # Convert inputs to numpy arrays
    pts = np.array(pts, dtype="float32")
    marker_coordinates = np.array(marker_coordinates)
    marker2_position = np.array(marker2_position, dtype="float32")
    
    # print("=" * 60)
    # print("START EXECUTION STEP BY STEP")
    # print("=" * 60)
    
    # ===== Step 1: Get the coordinates of marker2 (P3) and treat it as the new BR =====
    P3 = marker2_position  # Bottom-Right
    xP3, yP3 = P3[0], P3[1]
    
    # print(f"Step 1: Get marker2 coordinates as P3 (Bottom-Right)")
    # print(f"    P3 = ({xP3:.2f}, {yP3:.2f})")    
    # Find the 3 remaining marker1 points (excluding marker2)
    marker1_points = []
    for i, pt in enumerate(pts):
        # Compare with tolerance to avoid floating point errors
        if not (abs(pt[0] - marker2_position[0]) < 1e-6 and abs(pt[1] - marker2_position[1]) < 1e-6):
            marker1_points.append(pt)
    
    marker1_points = np.array(marker1_points, dtype="float32")
    # print(f"Remaining marker1 points: {len(marker1_points)} points")
    # for i, pt in enumerate(marker1_points):
    #     print(f"Marker1[{i}] = ({pt[0]:.2f}, {pt[1]:.2f})")
    
    # ===== Step 2: Find d1 - minimum distance from P3 to the 3 remaining points =====
    # print(f"\nStep 2: Find d1 - minimum distance from P3 to the 3 remaining points")
    
    distances = []
    for i, point in enumerate(marker1_points):
        dist = np.sqrt((P3[0] - point[0])**2 + (P3[1] - point[1])**2)
        distances.append(dist)
        # print(f"    Distance from P3 to Marker1[{i}]: {dist:.2f}")
    
    d1 = min(distances)
    closest_point_idx = np.argmin(distances)
    P4 = marker1_points[closest_point_idx]  # Bottom-Left (closest point to P3)
    
    # print(f"    d1 (minimum distance) = {d1:.2f}")
    # print(f"    Closest point to P3 selected as P4 = ({P4[0]:.2f}, {P4[1]:.2f})")
    
    # ===== Step 3: Derive the coordinates of P4 (bottom-left) =====
    # print(f"\nStep 3: P4 (Bottom-Left) has been determined")
    # print(f"    P4 = ({P4[0]:.2f}, {P4[1]:.2f})")
    
    # ===== Step 4: Determine the coordinates of P4' =====
    # print(f"\nStep 4: Determine the coordinates of P4' with xP4' = xP3 - d1, yP4' = yP3")
    
    xP4_prime = xP3 - d1
    yP4_prime = yP3
    P4_prime = np.array([xP4_prime, yP4_prime], dtype="float32")
    
    # print(f"    xP4' = xP3 - d1 = {xP3:.2f} - {d1:.2f} = {xP4_prime:.2f}")
    # print(f"    yP4' = yP3 = {yP4_prime:.2f}")
    # print(f"    P4' = ({xP4_prime:.2f}, {yP4_prime:.2f})")
    
    # ===== Step 5: Calculate the distance dP4P4' =====
    # print(f"\nStep 5: Calculate the distance dP4P4' from P4 to P4'")
    
    dP4P4_prime = np.sqrt((P4[0] - P4_prime[0])**2 + (P4[1] - P4_prime[1])**2)
    # print(f"    dP4P4' = sqrt(({P4[0]:.2f} - {P4_prime[0]:.2f})² + ({P4[1]:.2f} - {P4_prime[1]:.2f})²)")
    # print(f"    dP4P4' = {dP4P4_prime:.2f}")
    
    # ===== Step 6: Calculate angle alpha =====
    # print(f"\nStep 6: Calculate angle alpha = cos⁻¹((2*d1² - dP4P4'²) / (2*d1²))")
    
    numerator = 2 * d1**2 - dP4P4_prime**2
    denominator = 2 * d1**2
    
    # print(f"    Numerator = 2*d1² - dP4P4'² = 2*{d1:.2f}² - {dP4P4_prime:.2f}² = {numerator:.2f}")
    # print(f"    Denominator = 2*d1² = 2*{d1:.2f}² = {denominator:.2f}")
    
    if denominator != 0:
        cos_alpha = numerator / denominator
        # print(f"    cos(alpha) = {numerator:.2f} / {denominator:.2f} = {cos_alpha:.4f}")
        
        # Ensure cos_alpha is within the range [-1, 1]
        cos_alpha_clipped = np.clip(cos_alpha, -1, 1)
        # if cos_alpha != cos_alpha_clipped:
            # print(f"    ⚠️  cos(alpha) adjusted from {cos_alpha:.4f} to {cos_alpha_clipped:.4f}")
        
        alpha_radian = np.arccos(cos_alpha_clipped)
        alpha_degrees = np.degrees(alpha_radian)
        
        # Determine rotation direction by comparing P4.y vs P4'.y
        # In image coordinates (y increases downward):
        #   P4 is ABOVE P4' (P4.y < P4'.y) → image tilted clockwise → correct counter-clockwise → positive angle
        #   P4 is BELOW P4' (P4.y > P4'.y) → image tilted counter-clockwise → correct clockwise → negative angle
        if P4[1] > P4_prime[1]:
            alpha_degrees = -alpha_degrees
            direction = "Clockwise (negative)"
        else:
            direction = "Counter-clockwise (positive)"
        
        # print(f"    alpha = cos⁻¹({cos_alpha_clipped:.4f}) = {alpha_radian:.4f} radian = {alpha_degrees:.2f}°")
        # print(f"    P4.y={P4[1]:.2f} vs P4'.y={P4_prime[1]:.2f} → Direction: {direction}")
    else:
        # print("    ❌ Error: Denominator = 0, cannot calculate angle")
        alpha_degrees = 0
    
    # ===== Step 7: Prepare information for image rotation =====
    # print(f"\nStep 7: Information for image rotation")
    # print(f"    Rotation angle: {alpha_degrees:.2f}°")
    # print(f"    Direction: {'Counter-clockwise' if alpha_degrees > 0 else ('Clockwise' if alpha_degrees < 0 else 'No rotation')}")
    
    # Build rect with P3 fixed at bottom-right position
    remaining_points = []
    for point in marker1_points:
        if not np.array_equal(point, P4):
            remaining_points.append(point)
    
    # Sort rect in order: [TL, TR, BR, BL]
    if len(remaining_points) >= 2:
        P1 = remaining_points[0]  # Top-Left (tentative)
        P2 = remaining_points[1]  # Top-Right (tentative)
    else:
        # Fallback
        P1 = marker1_points[0] if len(marker1_points) > 0 else P3
        P2 = marker1_points[1] if len(marker1_points) > 1 else P3
    
    rect = [P1, P2, P3, P4]  # [TL, TR, BR, BL]

    
    # Calculate marker coordinates with offset
    marker_coordinates_true = []
    param = 0
    # print("rect", rect)
    marker_coordinates_true.append(calculate_new_coordinates(marker_coordinates, rect[0], -param, -param))
    marker_coordinates_true.append(calculate_new_coordinates(marker_coordinates, rect[1], param, -param))
    marker_coordinates_true.append(calculate_new_coordinates(marker_coordinates, rect[2], param, param))
    marker_coordinates_true.append(calculate_new_coordinates(marker_coordinates, rect[3], -param, param))
    
    marker_coordinates_true = np.array([marker_coordinates_true]).reshape(-1, 1, 2)
    # marker_coordinates_true = marker_coordinates_true.reshape(-1, 2).astype(int).tolist()
    # print(f"\nRESULT:")
    # print(f"    P1 (Top-Left): ({rect[0][0]:.2f}, {rect[0][1]:.2f})")
    # print(f"    P2 (Top-Right): ({rect[1][0]:.2f}, {rect[1][1]:.2f})")
    # print(f"    P3 (Bottom-Right): ({rect[2][0]:.2f}, {rect[2][1]:.2f}) ← Marker2")
    # print(f"    P4 (Bottom-Left): ({rect[3][0]:.2f}, {rect[3][1]:.2f})")
    # print(f"    Rotation angle: {alpha_degrees:.2f}°")
    # print("=" * 60)
    
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


#  Can cut answer and info images because after preprocessing, the image is almost in a fixed position and size
def crop_image_answer(img, numberAnswer):
    """
    Crop the three answer-column regions from a 1056×1500 document image,
    resize each to 250×640 for model inference.

    Returns:
        sorted_ans_blocks_resize : list of resized column images
        size_array               : list of original (width, height) per column
        coord_array              : list of (x, y) origins per column
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
