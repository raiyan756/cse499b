
from flask import Flask, request, jsonify, render_template
from pathlib import Path
import platform
import pathlib
import torch
from models.common import DetectMultiBackend
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_bytes
import tempfile
import os

# Windows fix
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

# Load YOLOv5 tick model
model_path = Path("tick_model.pt")
device = 'cpu'
model = DetectMultiBackend(model_path, device=device)
model.eval()

def run_ocr(image_path):
    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image, lang='eng')
    return text

def parse_mcq_text(raw_text):
    mcqs = []
    lines = raw_text.split('\n')
    current = {"question": "", "options": {}}
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        elif line[0].isdigit():
            if current["question"]:
                mcqs.append(current)
                current = {"question": "", "options": {}}
            current["question"] = line
        elif line[0] in ['A', 'B', 'C', 'D'] and len(line) > 2:
            current["options"][line[0]] = line[2:].strip()
    if current["question"]:
        mcqs.append(current)
    return mcqs

def run_tick_detection(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).float() / 255.0
    img = img.to(device)

    ticks = []
    with torch.no_grad():
        pred = model(img, augment=False, visualize=False)[0]
        if pred is not None and len(pred) > 0:
            pred = pred.cpu().numpy()
            for box in pred:
                coords = np.array(box[:4]).flatten()
                if coords.shape[0] != 4:
                    continue
                x1, y1, x2, y2 = coords.tolist()
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                ticks.append({
                    "x": float(x_center),
                    "y": float(y_center),
                    "w": float(w),
                    "h": float(h)
                })
    return ticks

def map_tick_to_option(tick_boxes, mcqs, image_path):
    image = cv2.imread(image_path)
    h_img, w_img = image.shape[:2]
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # Group OCR data by line number
    lines = {}
    for i in range(len(ocr_data['text'])):
        if ocr_data['text'][i].strip() == "":
            continue
        key = (ocr_data['block_num'][i], ocr_data['par_num'][i], ocr_data['line_num'][i])
        if key not in lines:
            lines[key] = {
                "text": [],
                "center_x": [],
                "center_y": []
            }
        lines[key]["text"].append(ocr_data['text'][i])
        x = ocr_data['left'][i] + ocr_data['width'][i] / 2
        y = ocr_data['top'][i] + ocr_data['height'][i] / 2
        lines[key]["center_x"].append(x)
        lines[key]["center_y"].append(y)

    # Compute closest line for each tick
    for tick in tick_boxes:
        tick_x = tick['x'] * (w_img / 640.0)
        tick_y = tick['y'] * (h_img / 640.0)

        best_line_text = ""
        best_distance = float('inf')

        for line in lines.values():
            avg_x = np.mean(line["center_x"])
            avg_y = np.mean(line["center_y"])
            distance = ((tick_x - avg_x) ** 2 + (tick_y - avg_y) ** 2) ** 0.5
            if distance < best_distance:
                best_distance = distance
                best_line_text = " ".join(line["text"]).strip()

        # Match line to an option
        for mcq in mcqs:
            for key, text in mcq["options"].items():
                if best_line_text and (best_line_text in text or text in best_line_text):
                    mcq["student_ticked"] = {
                        "option": key,
                        "text": text
                    }

    return mcqs

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/extract', methods=['POST'])
def extract_from_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files['file']
    if not uploaded_file.filename.endswith('.pdf'):
        return jsonify({"error": "Please upload a PDF file"}), 400

    with tempfile.TemporaryDirectory() as temp_dir:
        pages = convert_from_bytes(uploaded_file.read(), dpi=300)
        image_path = os.path.join(temp_dir, "page.jpg")
        pages[0].save(image_path, "JPEG")

        raw_text = run_ocr(image_path)
        structured_mcqs = parse_mcq_text(raw_text)
        tick_boxes = run_tick_detection(image_path)
        final_mcqs = map_tick_to_option(tick_boxes, structured_mcqs, image_path)

        return jsonify(final_mcqs)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
