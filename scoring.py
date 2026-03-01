import os
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import json
import shutil
from utils import (
    orient_image_step_by_step, generate_output,
    get_parameter_number_anwser, get_remainder,
    remove_elements_info, remove_elements_answer, remove_elements_marker,
    get_class_answer, get_class_info, get_class_marker,
    get_coordinates, get_coordinates_info,
    orient_image_by_angle, rotate_image_by_angle,
    warning_color, green_color, blue_color, threshold_warning,
    mergeImages, crop_image_answer, crop_image_info,
)


# ============================================ HANDLE MARKER =======================================

def get_marker(image, model, maybe_wrong_marker, folder_maybe_wrong):
    try:
        results = model.predict(image)
        data = results[0].boxes.data
        list_marker = []
        marker_coordinates = []
        validate_marker = []
        count_marker2 = 0
        count_maker1 = 0
        for i, data in enumerate(data):
            validate_marker.append(data)
        validate_marker = remove_elements_marker(validate_marker)
        for i, marker in enumerate(validate_marker):
            x1 = int(marker[0])
            y1 = int(marker[1])
            x2 = int(marker[2])
            y2 = int(marker[3])
            conf = round(float(marker[4]), 3)
            class_marker = int(marker[5])
            if (class_marker == 1):
                count_marker2 += 1
                marker2 = [x1, y1]
            if (class_marker == 0):
                count_maker1 += 1
            if (class_marker == 0 or class_marker == 1):
                list_marker.append([x1, y1])
                marker_coordinates.append([x1, y1, x2, y2])
            if (class_marker >= 0):
                cv2.rectangle(image, (x1, y1), (x2, y2), green_color if conf > threshold_warning else warning_color,
                    1 if conf > threshold_warning else 2)
                cv2.putText(image, str(get_class_marker(class_marker)) if conf > threshold_warning else str(f"{get_class_marker(class_marker)}-{conf}"),
                    (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4 if conf > threshold_warning else 0.5,
                    blue_color if conf > threshold_warning else warning_color, 1,cv2.LINE_AA)
     
        # Handle errors
        if count_marker2 != 1 or count_maker1 != 3:
            error_message = f"Image {filename} could not detect enough markers or may be missing corners"
            maybe_wrong_marker.append(error_message)
            raise Exception(error_message)

        marker_coordinates_true, alpha_degrees = orient_image_step_by_step(list_marker, marker_coordinates, marker2)
        rotated_image, rotation_matrix = rotate_image_by_angle(image, alpha_degrees)
    
        # ===== STEP 8 & 9: Crop image from the rotated image =====
        marker_coordinates_true = marker_coordinates_true.reshape(-1, 2).astype(int).tolist()
        # Apply rotation matrix to find new coordinates of the corners
        rotated_corners = []

        for corner in marker_coordinates_true:
            # Convert coordinates to homogeneous form [x, y, 1]
            point = np.array([corner[0], corner[1], 1])
            # Apply the rotation matrix obtained from rotate_image_by_angle
            rotated_point = rotation_matrix.dot(point)
            rotated_corners.append([int(rotated_point[0]), int(rotated_point[1])])

        # Crop the rotated image (rotated_image) using the new coordinates (rotated_corners)
        cropped_document = generate_output(rotated_image, rotated_corners)
        
        # Show the cropped image
        imgResize_cropped = cv2.resize(cropped_document, (506, 800), interpolation=cv2.INTER_AREA)

        return cropped_document, maybe_wrong_marker
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, maybe_wrong_marker
    

# ============================================ PREDICT IMAGE COLUMN ANSWER =======================================

def predictAnswer(img, model, index, numberAnswer):
    results = model.predict(img)
    data = results[0].boxes.data
    list_label = []
    for i, data in enumerate(data):
        list_label.append(data)
    list_label = sorted(list_label, key=lambda x: x[1])
    list_label = remove_elements_answer(list_label)
    array_answer = []
    maybe_wrong_answer = []
    for i, answer in enumerate(list_label):
        if index == get_parameter_number_anwser(numberAnswer) and i == get_remainder(numberAnswer):
            break
        class_answer = get_class_answer(int(answer[5]))
        array_answer.append(class_answer)
        x1 = int(answer[0])
        y1 = int(answer[1])
        x2 = int(answer[2])
        y2 = int(answer[3])
        conf = round(float(answer[4]), 3)
        class_answer = int(answer[5])
        if conf < threshold_warning:
            maybe_wrong_answer.append(f'[LOW CONF] Answer zone | File: {filename} | Question {index * 20 + i + 1} | Predicted: "{get_class_answer(class_answer)}" | Conf: {conf}')
            
        for char in str(get_class_answer(class_answer)):
            point1, point2, point3, point4 = get_coordinates(x1, y1, x2, y2, char)
            # Draw rectangle on unanswered labels only when conf < threshold_warning
            if char == "x":
                if conf < threshold_warning:
                    cv2.rectangle(img, (point1, point2), (point3, point4), warning_color, 2)
                    cv2.putText(img, str(f"{get_class_answer(class_answer)}-{conf}"),
                        (point1, point2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, warning_color, 1,cv2.LINE_AA)
            else:
                cv2.rectangle(img, (point1, point2), (point3, point4), green_color if conf > threshold_warning else warning_color,
                    1 if conf > threshold_warning else 2)
                cv2.putText(img, str(char) if conf > threshold_warning else str(f"{get_class_answer(class_answer)}-{conf}"),
                    (point1, point2), cv2.FONT_HERSHEY_SIMPLEX, 0.4 if conf > threshold_warning else 0.5,
                    blue_color if conf > threshold_warning else warning_color, 1,cv2.LINE_AA)
        img_graft = cv2.resize(img, (350, 896), interpolation=cv2.INTER_AREA)
        
    return array_answer, img_graft, maybe_wrong_answer


def predictInfo(img, model, filename):
    results = model.predict(img)
    data = results[0].boxes.data
    numberClassRecognition = len(data)
    list_label = []
    for i, data in enumerate(data):
        list_label.append(data)
    list_label = sorted(list_label, key=lambda x: x[0])
    list_label = remove_elements_info(list_label)
    dict_info = {}
    for i, info in enumerate(list_label):
        class_info = get_class_info(int(info[5]))
        dict_info[f"{i+1}"] = class_info
        x1 = int(info[0])
        y1 = int(info[1])
        x2 = int(info[2])
        y2 = int(info[3])
        conf = round(float(info[4]), 3)
        class_info = int(info[5])
        if conf < threshold_warning:
            maybe_wrong_info.append(f'[LOW CONF] Info zone | File: {filename} | Column {i + 1} (left→right) | Predicted: "{get_class_info(class_info)}" | Conf: {conf}')

        point1, point2, point3, point4 = get_coordinates_info(x1, y1, x2, y2, get_class_info(class_info))
        cv2.rectangle(img, (point1, point2), (point3, point4), 
            green_color if conf > threshold_warning else warning_color, 1 if conf > threshold_warning else 2)
        cv2.putText(img, str(get_class_info(class_info)) if conf > threshold_warning else str(f"{get_class_info(class_info)}-{conf}"), 
            (point1, point2),cv2.FONT_HERSHEY_SIMPLEX,
            0.4 if conf > threshold_warning else 0.5, blue_color if conf > threshold_warning else warning_color, 1 ,cv2.LINE_AA,)
    if numberClassRecognition > 5:
        class_code = "".join([dict_info[str(i)] if str(i) not in dict_info or dict_info[str(i)] != "unchoice" else "x" for i in range(1, 7)])
        student_code = "".join(list(dict_info.values())[6:-3])
        exam_code = "".join(list(dict_info.values())[-3:])
        result_info = { "class_code": class_code, "student_code": student_code, "exam_code": exam_code}
        imgResize = cv2.resize(img, (506, 500), interpolation=cv2.INTER_AREA)
    elif numberClassRecognition <= 5:
        result_info = { "class_code": "", "student_code": "", "exam_code": ""}
        imgResize = cv2.resize(img, (506, 500), interpolation=cv2.INTER_AREA)

    return result_info, imgResize, numberClassRecognition, maybe_wrong_info



if __name__ == "__main__":
    # ========================== Measure execution time ====================================
    # start_time = time.time()
    # ===================== Declare and load models ==============================
    pWeight_marker = "./Model/marker.pt"
    pWeight_answer = "./Model/answer.pt"
    pWeight_info = "./Model/info.pt"
    model_info = YOLO(pWeight_info)
    model_marker = YOLO(pWeight_marker)
    model_answer = YOLO(pWeight_answer)
    # ======================= Declare command-line arguments ===============================
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("input", help="input")
    args = parser.parse_args()

    # ================= Create folders for original and processed images ===============================
    folder_path = f"images/answer_sheets/{args.input}"
    folder_path_handle = f"images/answer_sheets/{args.input}/HandledSheets"
    folder_scored_path = f"images/answer_sheets/{args.input}/ScoredSheets"
    folder_maybe_wrong = f"images/answer_sheets/{args.input}/MayBeWrong"
    if os.path.exists(folder_maybe_wrong):
        shutil.rmtree(folder_maybe_wrong)
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
        except OSError:
            print(f"Error: Could not create directory {folder_path}.")
            
    if not os.path.exists(folder_path_handle):
        try:
            os.makedirs(folder_path_handle)
        except OSError:
            print(f"Error: Could not create directory {folder_path_handle}.")
            
    if not os.path.exists(folder_scored_path):
        try:
            os.makedirs(folder_scored_path)
        except OSError:
            print(f"Error: Could not create directory {folder_scored_path}.")
            
    if not os.path.exists(folder_maybe_wrong):
        try:
            os.makedirs(folder_maybe_wrong)
        except OSError:
            print(f"Error: Could not create directory {folder_maybe_wrong}.")            
            
    maybe_wrong_info = []
    maybe_wrong_answer_array = []
    maybe_wrong_marker = []


    # ================================= Main program =================================
    file_names = os.listdir(folder_path)
    file_names.sort()
    for filename in file_names:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            document, maybe_wrong_marker = get_marker(image, model_marker, maybe_wrong_marker,folder_maybe_wrong)
            if (document is None):
                continue
            
            # With the new logic, the image is already rotated and marker2 is always at bottom-right
            # Just resize to standard dimensions, no additional rotation needed
            document = cv2.resize(document, (1056, 1500), interpolation=cv2.INTER_AREA)
            document = document / 255
            # ========================== Crop student ID and exam code area ===============================
            img_resize = crop_image_info(document)
            result_info, imgResize, numberClassRecognition, maybe_wrong_info = predictInfo(img=img_resize, model=model_info, filename=filename)
            numberAnswer = 60
            # =================================== Get answers ==============================
            result_answer, size_array, coord_array = crop_image_answer(cv2.convertScaleAbs(document * 255), numberAnswer)
            list_answer = []
            array_img_graft = []
            for i, answer in enumerate(result_answer):
                selected_answer, img_graft, maybe_wrong_answer = predictAnswer(img=answer, model=model_answer, index=i, numberAnswer=numberAnswer)
                list_answer = list_answer + selected_answer
                array_img_graft.append(img_graft)
                maybe_wrong_answer_array += maybe_wrong_answer
            # ================================= Format JSON output =============================
            array_result = []
            for key, value in enumerate(list_answer):
                item = {"questionNo": int(key) + 1, "selectedAnswers": value}
                array_result.append(item)
                if key == (numberAnswer - 1):
                    break
               
            if len(result_info) == 3:
                result = {
                    "examClassCode": result_info["class_code"],
                    "studentCode": result_info["student_code"],
                    "testSetCode": result_info["exam_code"],
                    "answers": array_result
                }
            
            # ============================= Merge images =====================================
            handled_scored_img = mergeImages(filename, coord_array, array_img_graft, background_image=document, imgInfo=imgResize, args=args)
            result["handledScoredImg"] = handled_scored_img
            result["originalImg"] = image_path
            result["originalImgFileName"] = filename
                
            # =============================== Write JSON file ==========================

            orig_file_name = filename.split(".")[0]
            file_path = f"{folder_scored_path}/{orig_file_name}_data.json"
            dir_path = os.path.dirname(file_path)

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            # Write dictionary data to JSON file
            with open(file_path, "w") as file:
                json.dump(result, file)
            # =============================== Write warning file for potentially incorrect results ==========================

        # ========================================= Measure execution time ==========================
        # print("Execution time: ", time.time() - start_time, " seconds")
    if len(maybe_wrong_info) > 0 or len(maybe_wrong_answer_array) > 0 or len(maybe_wrong_marker) > 0:
        with open(f"{folder_maybe_wrong}/maybe_wrong.txt", "w", encoding="utf-8") as f:
            if len(maybe_wrong_marker) > 0:
                for string in maybe_wrong_marker:
                    f.write(string + "\n")
            if len(maybe_wrong_info) > 0:
                for string in maybe_wrong_info:
                    f.write(string + "\n")
            if len(maybe_wrong_answer_array) > 0:
                for string in maybe_wrong_answer_array:
                    f.write(string + "\n")
      # Reset data 
    maybe_wrong_marker = []
    maybe_wrong_info = []
    maybe_wrong_answer_array = []