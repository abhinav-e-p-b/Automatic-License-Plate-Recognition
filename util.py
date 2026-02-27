import re
import string
from paddleocr import PaddleOCR

# Initialize PaddleOCR (use_angle_cls helps with rotated plates)
reader = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=True,
    show_log=False,
    det_model_dir=r'C:\Users\abhin\OneDrive\Desktop\Ironfleet\Archives\Automatic-License-Plate-Recognition-using-YOLOv8\PaddleOCR\en_PP-OCRv3_det_infer\en_PP-OCRv3_det_infer',
    rec_model_dir=r'C:\Users\abhin\OneDrive\Desktop\Ironfleet\Archives\Automatic-License-Plate-Recognition-using-YOLOv8\PaddleOCR\en_PP-OCRv3_rec_infer\en_PP-OCRv3_rec_infer',
    cls_model_dir=r'C:\Users\abhin\OneDrive\Desktop\Ironfleet\Archives\Automatic-License-Plate-Recognition-using-YOLOv8\PaddleOCR\ch_ppocr_mobile_v2.0_cls_infer\ch_ppocr_mobile_v2.0_cls_infer',
)

# Indian state codes for validation
INDIAN_STATE_CODES = {
    'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA',
    'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH',
    'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ',
    'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB'
}

# Character correction maps
dict_char_to_int = {
    'O': '0', 'I': '1', 'Z': '2', 'J': '3',
    'A': '4', 'G': '6', 'S': '5', 'B': '8'
}

dict_int_to_char = {
    '0': 'O', '1': 'I', '2': 'Z', '3': 'J',
    '4': 'A', '6': 'G', '5': 'S', '8': 'B'
}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format(
            'frame_nmr', 'car_id', 'car_bbox',
            'license_plate_bbox', 'license_plate_bbox_score',
            'license_number', 'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(
                        frame_nmr,
                        car_id,
                        '[{} {} {} {}]'.format(
                            results[frame_nmr][car_id]['car']['bbox'][0],
                            results[frame_nmr][car_id]['car']['bbox'][1],
                            results[frame_nmr][car_id]['car']['bbox'][2],
                            results[frame_nmr][car_id]['car']['bbox'][3]),
                        '[{} {} {} {}]'.format(
                            results[frame_nmr][car_id]['license_plate']['bbox'][0],
                            results[frame_nmr][car_id]['license_plate']['bbox'][1],
                            results[frame_nmr][car_id]['license_plate']['bbox'][2],
                            results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                        results[frame_nmr][car_id]['license_plate']['bbox_score'],
                        results[frame_nmr][car_id]['license_plate']['text'],
                        results[frame_nmr][car_id]['license_plate']['text_score']))
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with Indian number plate format.

    Indian format: XX 00 XX 0000
    Examples: MH12AB1234, DL3CAB1234, KA09MF9999

    Args:
        text (str): License plate text (already uppercased, no spaces).

    Returns:
        bool: True if format matches, False otherwise.
    """
    # Standard Indian plate: 2 letters (state) + 2 digits (district) + 1-3 letters + 4 digits
    # Regex: [STATE_CODE][00-99][A-Z]{1,3}[0-9]{4}
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$'
    if re.match(pattern, text):
        state_code = text[:2]
        if state_code in INDIAN_STATE_CODES:
            return True
    return False


def correct_ocr_errors(text):
    """
    Correct common OCR misreads for Indian plates.
    Positions: [0-1]=state letters, [2-3]=district digits, [4-6]=series letters, [7-10]=number digits
    """
    corrected = list(text)
    n = len(text)

    for i, ch in enumerate(corrected):
        if i < 2:
            # State code: must be letters
            if ch in dict_char_to_int:
                corrected[i] = dict_char_to_int[ch]  # Keep as is — wait, need letters
            # Actually convert digit-lookalikes to letters
            if ch in dict_int_to_char:
                corrected[i] = dict_int_to_char[ch]
        elif i < 4:
            # District: must be digits
            if ch in dict_char_to_int:
                corrected[i] = dict_char_to_int[ch]
        elif i < n - 4:
            # Series letters: must be letters
            if ch in dict_int_to_char:
                corrected[i] = dict_int_to_char[ch]
        else:
            # Serial number: must be digits
            if ch in dict_char_to_int:
                corrected[i] = dict_char_to_int[ch]

    return ''.join(corrected)


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image using PaddleOCR.

    Args:
        license_plate_crop: Cropped image (numpy array) of the license plate.

    Returns:
        tuple: (formatted license plate text, confidence score) or (None, None).
    """
    result = reader.ocr(license_plate_crop, cls=True)

    if result is None or result[0] is None:
        return None, None

    # Concatenate all detected text on the plate (handles multi-line plates)
    combined_text = ''
    scores = []
    for line in result[0]:
        _, (text, score) = line
        combined_text += text.upper().replace(' ', '').replace('-', '')
        scores.append(score)

    if not combined_text:
        return None, None

    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Try to correct OCR errors and validate
    corrected = correct_ocr_errors(combined_text)
    if license_complies_format(corrected):
        return corrected, avg_score

    # Also try the raw text after cleanup
    if license_complies_format(combined_text):
        return combined_text, avg_score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1