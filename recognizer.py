import sys, os
import torch
import tempfile
import subprocess
import logging
import re
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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


model_name = "IlyaGusev/saiga_llama3_8b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer_llm = AutoTokenizer.from_pretrained(model_name, use_fast=False)

model_llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)


def split_into_sentences(text):
    sentence_endings = r'(?<=[.!?…])\s+(?=[А-ЯЁ])'
    sentences = re.split(sentence_endings, text.strip())
    return [s.strip() for s in sentences if s.strip()]

def correct_large_text_with_llm(model, tokenizer, full_text, max_total_tokens=4096, max_response_ratio=0.6):
    sentences = split_into_sentences(full_text)
    corrected_sentences = []

    for sent in sentences:
        prompt = f"""Исправь орфографические и грамматические ошибки в тексте. Верни только исправленную версию,
         без добавления новых слов, комментариев, пояснений, ссылок или временных отметок.

Оригинал: {sent}
Исправленный текст:"""

        # Токенизация без генерации, чтобы оценить длину
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        input_len = input_ids.shape[1]

        # Оставляем запас для генерации
        available_tokens = max_total_tokens - input_len
        max_new_tokens = min(int(input_len * max_response_ratio), available_tokens)
        if max_new_tokens <= 0:
            logging.warning(f"[WARNING] Недостаточно места для генерации по предложению: {sent}")
            max_new_tokens = 32  # fallback

        # Генерация
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.5,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        corrected = output_text.split("Исправленный текст:")[-1].strip()
        corrected = corrected.split("\n")[0].strip()

        # Удаляем шаблонные фразы
        ban_phrases = ["это правильный ответ", "именно поэтому", "начните", "в заключение"]
        if any(bad in corrected.lower() for bad in ban_phrases):
            corrected = corrected.split(".")[0].strip() + "."

        corrected_sentences.append(corrected)

    return " ".join(corrected_sentences)


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
        image = image_pil.convert("RGB")
        image.save(image_path, format="JPEG")
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

        imgs = [
            resize_with_aspect_and_padding(
                Image.open(os.path.join(cropped_dir, f"cropped_image{i + 1}.jpg")).convert("RGB"))
            for i in range(len(sorted_coords))
        ]

        pixel_values = processor(images=imgs, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(
            pixel_values,
            max_length=256,
            num_beams=1,
            do_sample=False
        )

        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        recognized_text = " ".join(texts).strip()

        corrected_text = correct_large_text_with_llm(model_llm, tokenizer_llm, recognized_text)

        return recognized_text, corrected_text, 'ok'