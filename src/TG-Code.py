import telebot
import os
import io
import sys
import torch
import shutil
import tempfile
import crop  
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import language_tool_python
 

cwd = os.getcwd()
root_dir = os.path.dirname(cwd)
parent_root_dir = os.path.dirname(root_dir)
yolo_dir = os.path.join(parent_root_dir, "yolo_v5", "yolov5")
YOLO_WEIGHTS = os.path.join(root_dir, "models", "yolov5", "best.pt")
TROCR_MODEL_DIR = os.path.join(root_dir, "models", "trocr", "v3", "trocr_model")
TROCR_PROCESSOR_DIR = os.path.join(root_dir, "models", "trocr", "v3", "trocr_processor")

sys.path.append(yolo_dir)

from detect import run 

bot = telebot.TeleBot('7654203891:AAFEb7yBUe5YqoP4ADJnl8Ipa7GzJlJjvt4') 
processor = TrOCRProcessor.from_pretrained(TROCR_PROCESSOR_DIR)
model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")

def clear_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def convert_to_jpeg(image_pil, output_path):
    image_pil.convert("RGB").save(output_path, format="JPEG")
    return output_path

def check_spelling(text):
    tool = language_tool_python.LanguageTool('ru-RU')
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)

    error_log = ""
    for match in matches:
        error_log += f"• {match.message}\n  ⤷ {match.context.strip()}\n"

    tool.close()
    return corrected_text, error_log

def process_image_pipeline(image_pil):
    with tempfile.TemporaryDirectory() as temp_dir:
        original_path = os.path.join(temp_dir, "input.jpg")
        bbox_dir = os.path.join(temp_dir, "bbox", "labels", "input.txt")
        cropped_dir = os.path.join(temp_dir, "crops")
        project_dir = os.path.join(temp_dir, "bbox")

        convert_to_jpeg(image_pil, original_path)

        run(
            weights=YOLO_WEIGHTS,
            source=original_path,
            conf_thres=0.25,
            save_txt=True,
            save_crop=False,
            project=project_dir,
            name='',
            exist_ok=False
        )

        jpg_for_size = Image.open(original_path).convert("RGB")
        width, height = jpg_for_size.size
        label_path = os.path.join(project_dir, "labels", "input.txt")
        coords = crop.read_coords(label_path)
        pixel_coords = crop.convert_to_pixel_coords(coords, width, height)
        sorted_coords = crop.sort_coords(pixel_coords)

        os.makedirs(cropped_dir, exist_ok=True)
        crop.crop_and_save_images(original_path, sorted_coords, cropped_dir)

        recognized_text = ""
        for i in range(len(sorted_coords)):
            path = os.path.join(cropped_dir, f"cropped_image{i+1}.jpg")
            try:
                img = Image.open(path).convert("RGB")
                pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(model.device)
                generated_ids = model.generate(pixel_values)
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                recognized_text += text + " "
            except Exception as e:
                print(f"Ошибка при обработке {path}: {e}")

        corrected, errors = check_spelling(recognized_text.strip())
        return recognized_text.strip(), corrected.strip(), errors


@bot.message_handler(content_types=['text'])
def handle_text(message):
    if message.text == "/start":
        bot.send_message(message.chat.id,
            "👋 Привет! Я бот, который умеет распознавать *русский рукописный текст* с изображений.\n\n"
            "📸 Просто отправь мне фото, и я постараюсь извлечь текст, распознать его и даже проверить орфографию!\n\n"
            "Для справки напиши /help.")
    elif message.text == "/help":
        bot.send_message(message.chat.id,
            "🆘 *Помощь по боту*\n\n"
            "Вот что я умею:\n"
            "1️⃣ Распознаю *рукописный текст* с изображений\n"
            "2️⃣ Проверяю *орфографию и грамматику*\n"
            "3️⃣ Даю исправленный текст\n\n"
            "📌 *Советы:*\n"
            "— Делай чёткие фото\n"
            "— Лучше, если на изображении только один текстовый блок\n"
            "📥 Просто отправь фото, и я начну работать!")
    else:
        bot.send_message(message.chat.id,
            "🤔 Прости, я не понимаю тебя...\nНапиши /help, чтобы узнать, что я умею.")


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        bot.send_message(message.chat.id, "📤 Получено изображение. Загружаю...")

        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded = bot.download_file(file_info.file_path)
        image = Image.open(io.BytesIO(downloaded)).convert("RGB")

        bot.send_message(message.chat.id, "🧠 Запускаю нейросети для распознавания...")
        raw_text, corrected_text, errors = process_image_pipeline(image)

        bot.send_message(message.chat.id, f"📝 *Распознанный текст:*\n`{raw_text}`", parse_mode="Markdown")
        bot.send_message(message.chat.id, f"✅ *Исправленный текст:*\n`{corrected_text}`", parse_mode="Markdown")

        if errors:
            bot.send_message(message.chat.id, f"🔍 *Обнаружены ошибки:*\n{errors}")
        else:
            bot.send_message(message.chat.id, "🎉 Ошибок не найдено!")

    except Exception as e:
        bot.send_message(message.chat.id, "❌ Произошла ошибка при обработке изображения. Попробуйте ещё раз.")
        print(f"Ошибка: {e}")


print("🚀 Бот запущен. Ожидаю изображения...")
bot.polling(none_stop=True)

