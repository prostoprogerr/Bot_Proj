import sys, os
import torch
import tempfile
import subprocess
import logging
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from bot_utils.check_spelling import check_spelling_and_grammar
from bot_utils import crop
from bot_utils.resize import resize_with_aspect_and_padding


root_dir = os.getcwd()
parent_root_dir = os.path.dirname(root_dir)
yolo_dir = os.path.join(parent_root_dir, "yolo_v5", "yolov5")
yolo_weights = os.path.join(root_dir, "models", "yolov5", "best.pt")
model_dir = os.path.join(root_dir, "models", "trocr", "v3", "model")
processor_dir = os.path.join(root_dir, "models", "trocr", "v3", "processor")

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = TrOCRProcessor.from_pretrained(processor_dir)
model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)

def run_yolo_subprocess(image_path, output_dir, yolo_weights):
    python_executable = sys.executable

    command = [
        python_executable, "detect.py",
        "--weights", yolo_weights,
        "--source", image_path,
        "--conf", "0.69",
        "--save-txt",
        "--save-conf",
        "--project", output_dir,
        "--name", "result",
        "--exist-ok"
    ]
    try:
        result = subprocess.run(
            command,
            cwd=yolo_dir,
            check=True,
            capture_output=True,
            text=True
        )
        logging.info("[YOLO STDOUT]: %s", result.stdout)
        logging.debug("[YOLO STDERR]: %s", result.stderr)
    except subprocess.CalledProcessError as e:
        logging.error("[ERROR] YOLOv5 subprocess failed:")
        logging.error("STDOUT: %s", e.stdout)
        logging.error("STDERR: %s", e.stderr)

def convert_to_jpeg(image_pil, output_path):
    image_pil.convert("RGB").save(output_path, format="JPEG")
    return output_path


def process_image_pipeline(image_pil):
    with tempfile.TemporaryDirectory() as base_dir:
        image_dir = os.path.join(base_dir, "input_images")
        bbox_dir = os.path.join(base_dir, "bbox")
        cropped_dir = os.path.join(base_dir, "crops")

        for path in [image_dir, bbox_dir, cropped_dir]:
            os.makedirs(path, exist_ok=True)

        image_path = os.path.join(image_dir, "input.jpg")
        convert_to_jpeg(image_pil, image_path)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        run_yolo_subprocess(image_path, bbox_dir, yolo_weights)

        label_path = os.path.join(bbox_dir, "result", "labels", "input.txt")
        if not os.path.exists(label_path):
            logging.error(f"[ERROR] Файл меток не найден: {label_path}")
            return "", "", "⚠️ Не удалось распознать текст: YOLO не нашёл текст на изображении."

        normalized_coords = crop.read_coords(label_path)
        pixel_coords = crop.convert_to_pixel_coords(normalized_coords, width, height)
        sorted_coords = crop.sort_coords(pixel_coords)
        crop.crop_and_save_images(image_path, sorted_coords, cropped_dir)

        recognized_text = ""
        for i in range(len(sorted_coords)):
            cropped_path = os.path.join(cropped_dir, f"cropped_image{i + 1}.jpg")
            try:
                img = resize_with_aspect_and_padding(Image.open(cropped_path).convert("RGB"))
                pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
                generated_ids = model.generate(pixel_values)
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                recognized_text += text + " "
            except Exception as e:
                logging.error(f"[ERROR] Ошибка при обработке {cropped_path}: {e}")

        corrected_text, errors = check_spelling_and_grammar(recognized_text.strip())

        return recognized_text.strip(), corrected_text.strip(), errors