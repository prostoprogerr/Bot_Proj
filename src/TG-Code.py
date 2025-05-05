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
yolo_weights = os.path.join(root_dir, "models", "yolov5", "best.pt")
model_dir = os.path.join(root_dir, "models", "trocr", "v3", "trocr_model")
processor_dir = os.path.join(root_dir, "models", "trocr", "v3", "trocr_processor")

sys.path.append(yolo_dir)

from detect import run

bot = telebot.TeleBot('7654203891:AAFEb7yBUe5YqoP4ADJnl8Ipa7GzJlJjvt4')
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = TrOCRProcessor.from_pretrained(processor_dir)
model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)


def clear_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def clear(image_dir, bbox_dir, cropped_dir):
    clear_directory(image_dir)
    clear_directory(bbox_dir)
    clear_directory(cropped_dir)

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

        run(
            weights=yolo_weights,
            source=image_path,
            conf_thres=0.7,
            save_txt=True,
            save_crop=False,
            project=bbox_dir,
            name='',
            exist_ok=True
        )

        label_path = os.path.join(bbox_dir, "labels", "input.txt")
        if not os.path.exists(label_path):
            print(f"[ERROR] Файл меток не найден: {label_path}")
            return "", "", "⚠️ Не удалось распознать текст: YOLO не нашёл текст на изображении."

        normalized_coords = crop.read_coords(label_path)
        pixel_coords = crop.convert_to_pixel_coords(normalized_coords, width, height)
        sorted_coords = crop.sort_coords(pixel_coords)
        crop.crop_and_save_images(image_path, sorted_coords, cropped_dir)

        recognized_text = ""
        for i in range(len(sorted_coords)):
            cropped_path = os.path.join(cropped_dir, f"cropped_image{i + 1}.jpg")
            try:
                img = Image.open(cropped_path).convert("RGB")
                pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
                generated_ids = model.generate(pixel_values)
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                recognized_text += text + " "
            except Exception as e:
                print(f"[ERROR] Ошибка при обработке {cropped_path}: {e}")

        corrected_text, errors = check_spelling(recognized_text.strip())

        return recognized_text.strip(), corrected_text.strip(), errors


@bot.message_handler(content_types=['text'])
def handle_text(message):
    if message.text == "/start":
        bot.send_message(message.chat.id,
                         "👋 Привет! Я бот, который умеет распознавать *русский рукописный текст* с изображений.\n\n"
                         "📸 Просто отправь мне фото, я извлеку текст, распознаю его и проверю орфографию!\n\n"
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
                         "— Лучше загружай изображение-документ (без сжатия)\n"
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

        bot.send_message(message.chat.id, f"📝 *Распознанный текст:*\n`{raw_text}")
        bot.send_message(message.chat.id, f"✅ *Исправленный текст:*\n`{corrected_text}")

        if errors:
            bot.send_message(message.chat.id, f"🔍 *Обнаружены ошибки:*\n{errors}")
        else:
            bot.send_message(message.chat.id, "🎉 Ошибок не найдено!")

    except Exception as e:
        bot.send_message(message.chat.id, "❌ Произошла ошибка при обработке изображения. Попробуйте ещё раз.")
        print(f"Ошибка: {e}")

@bot.message_handler(content_types=['document'])
def handle_image_document(message):
    try:
        file_name = message.document.file_name.lower()
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

        if not file_name.endswith(image_extensions):
            bot.send_message(message.chat.id, "⚠️ Пожалуйста, отправьте изображение (JPG, PNG и т.д.), а не другой тип файла.")
            return

        bot.send_message(message.chat.id, "📥 Получено изображение-документ. \n🧠 Распознаю текст...")

        file_info = bot.get_file(message.document.file_id)
        downloaded = bot.download_file(file_info.file_path)
        image = Image.open(io.BytesIO(downloaded)).convert("RGB")

        raw_text, corrected_text, errors = process_image_pipeline(image)

        bot.send_message(message.chat.id, f"📝 *Распознанный текст:*\n{raw_text}", parse_mode="Markdown")
        bot.send_message(message.chat.id, f"✅ *Исправленный текст:*\n{corrected_text}", parse_mode="Markdown")

        if errors:
            bot.send_message(message.chat.id, f"🔍 *Обнаружены ошибки:*\n{errors}")
        else:
            bot.send_message(message.chat.id, "🎉 Ошибок не найдено!")

    except Exception as e:
        bot.send_message(message.chat.id, "❌ Произошла ошибка при обработке изображения-документа.")
        print(f"[ERROR] Ошибка с изображением-документом: {e}")


print("🚀 Бот запущен. Ожидаю изображения...")
bot.polling(none_stop=True)