import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import random
import io

try:
    model = YOLO("best.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    model = YOLO("yolov8n.pt")

class_names = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'Alef', 'Be', 'Te', 'Se', 'Jim', 'Dal', 'Sin', 'Shin',
    'Sad', 'Ta', 'Za', 'Eyn', 'Ghaf', 'Lam', 'Mim', 'Nun',
    'He', 'Vav', 'Pe', 'Zhe', 'plate', 'Ye', 'Ze'
]

char_map = {
    0: '۰', 1: '۱', 2: '۲', 3: '۳', 4: '۴', 5: '۵', 6: '۶', 7: '۷', 8: '۸', 9: '۹',
    10: 'الف', 11: 'ب', 12: 'ت', 13: 'ث', 14: 'ج', 15: 'د', 16: 'س', 17: 'ش',
    18: 'ص', 19: 'ط', 20: 'ظ', 21: 'ع', 22: 'ق', 23: 'ل', 24: 'م', 25: 'ن',
    26: 'ه', 27: 'و', 28: 'پ', 29: 'ژ', 31: 'ی', 32: 'ز'
}

app = Flask(__name__)

def detect_plate_format(image: Image.Image):
    results = model(image, verbose=False)[0]
    recognized_characters = []

    if results.boxes is not None:
        boxes = results.boxes.data.cpu().numpy()
        for box in boxes:
            cls_id = int(box[5])
            if cls_id in char_map:
                recognized_characters.append({'box': box, 'char': char_map[cls_id]})

    if not recognized_characters:
        return None

    sorted_chars = sorted(recognized_characters, key=lambda item: item['box'][0])
    chars = [item['char'] for item in sorted_chars]

    digits = [c for c in chars if c.isdigit() or c in '۰۱۲۳۴۵۶۷۸۹']
    letters = [c for c in chars if c not in digits]

    if len(digits) >= 5 and letters:
        plate_parts = {
            "left_digits": "".join(digits[:2]),
            "letter": letters[0],
            "right_digits": "".join(digits[2:5]),
            "city_digits": "".join(digits[5:7]) if len(digits) > 5 else ""
        }
    else:
        plate_parts = {
            "left_digits": "",
            "letter": "",
            "right_digits": "".join(chars),
            "city_digits": ""
        }

    return plate_parts

@app.route("/detect_plate", methods=["POST"])
def detect_plate():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    plate_text = detect_plate_format(image)

    if plate_text is None:
        return jsonify({"message": "پلاک موجود نیست"})
    
    return jsonify({"plate_text": plate_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
