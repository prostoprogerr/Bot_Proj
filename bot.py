import io
import logging
import threading
import requests
import time
import os
import telebot
from dotenv import load_dotenv
from PIL import Image
from telebot.types import ReplyKeyboardMarkup, KeyboardButton

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

user_data = {}

load_dotenv()

bot = telebot.TeleBot(os.getenv('bot'))

def send_image_to_pipeline(image: Image.Image):
    with io.BytesIO() as output:
        image.save(output, format="JPEG")
        output.seek(0)
        files = {"file": ("image.jpg", output, "image/jpeg")}
        try:
            response = requests.post("http://localhost:8000/process/", files=files)
            response.raise_for_status()
            data = response.json()
            return data["recognized_text"], data["corrected_text"], data["errors"]
        except requests.RequestException as e:
            logging.error(f"[ERROR] Ошибка запроса к микросервису: {e}")
            raise

def show_typing(bot, chat_id, stop_event):
    while not stop_event.is_set():
        bot.send_chat_action(chat_id, action='typing')
        time.sleep(1.5)

def start_keyboard():
    markup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    start_button = KeyboardButton("Старт")
    help_button = KeyboardButton("Помощь")
    markup.add(start_button, help_button)
    return markup

def text_action_keyboard():
    markup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    get_text = KeyboardButton("Получить распознанный текст")
    get_corrected_text = KeyboardButton("Получить исправленный текст")
    get_all = KeyboardButton("Получить все сразу")
    markup.add(get_text)
    markup.add(get_corrected_text)
    markup.add(get_all)
    return markup

@bot.message_handler(content_types=['text'])
def handle_text(message):
    if message.text.lower() == "старт":
        bot.send_message(message.chat.id,
                         "👋 Привет! Я бот, который умеет распознавать *русский рукописный текст* с изображений.\n\n"
                         "📸 Просто отправь мне фото, я извлеку текст, распознаю его и проверю орфографию, грамматику и стиль.\n\n"
                         "Для справки нажми *Помощь*.",
                         reply_markup=start_keyboard(),
                         parse_mode="Markdown")

    elif message.text.lower() == "помощь":
        bot.send_message(message.chat.id,
                         "🆘 *Помощь по боту*\n\n"
                         "Вот что я умею:\n"
                         "1️⃣ Распознаю *рукописный текст* с изображений\n"
                         "2️⃣ Проверяю *орфографию, грамматику и стиль написания*\n"
                         "3️⃣ Даю исправленный текст\n\n"
                         "📌 Советы:\n"
                         "— Делай чёткие фото\n"
                         "— Лучше загружай изображение-документ (без сжатия)\n"
                         "📥 Просто отправь фото, и я начну работать!",
                         reply_markup=start_keyboard(),
                         parse_mode="Markdown")

    elif message.text == "Получить распознанный текст":
        data = user_data.get(message.chat.id)
        if data:
            bot.send_message(message.chat.id,
                             f"📝 *Распознанный текст:*\n```\n{data['raw_text']}\n```",
                             parse_mode="Markdown")
        else:
            bot.send_message(message.chat.id, "❌ Нет данных. Сначала отправьте изображение.")

    elif message.text == "Получить исправленный текст":
        data = user_data.get(message.chat.id)
        if data:
            bot.send_message(message.chat.id,
                             f"✅ *Исправленный текст:*\n```\n{data['corrected_text']}\n```",
                             parse_mode="Markdown")
        else:
            bot.send_message(message.chat.id, "❌ Нет данных. Сначала отправьте изображение.")

    elif message.text == "Получить все сразу":
        data = user_data.get(message.chat.id)
        if data:
            bot.send_message(message.chat.id,
                             f"📝 *Распознанный текст:*\n```\n{data['raw_text']}\n```",
                             parse_mode="Markdown")
            bot.send_message(message.chat.id,
                             f"✅ *Исправленный текст:*\n```\n{data['corrected_text']}\n```",
                             parse_mode="Markdown")
            if data["errors"]:
                bot.send_message(message.chat.id,
                                 f"🔍 *Обнаружены ошибки:*\n{data['errors']}",
                                 parse_mode="HTML")
            else:
                bot.send_message(message.chat.id,
                                 "🎉 Ошибок не найдено!")
        else:
            bot.send_message(message.chat.id, "❌ Нет данных. Сначала отправьте изображение.")

    else:
        bot.send_message(message.chat.id,
                         "🤔 Прости, я не понимаю тебя...\nНажми *Помощь*, чтобы узнать, что я умею.",
                         reply_markup=start_keyboard(),
                         parse_mode="Markdown")


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    stop_typing = threading.Event()
    typing_thread = threading.Thread(target=show_typing, args=(bot, message.chat.id, stop_typing))
    typing_thread.start()

    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded = bot.download_file(file_info.file_path)
        image = Image.open(io.BytesIO(downloaded)).convert("RGB")

        raw_text, corrected_text, errors = send_image_to_pipeline(image)

        user_data[message.chat.id] = {
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "errors": errors
        }

        stop_typing.set()
        typing_thread.join()

        bot.send_message(message.chat.id,
                         "✅ Изображение успешно обработано.\nЧто вы хотите сделать дальше?",
                         reply_markup=text_action_keyboard(),
                         parse_mode="Markdown")

    except Exception as e:
        stop_typing.set()
        typing_thread.join()
        bot.send_message(message.chat.id,
                         "❌ Произошла ошибка при обработке изображения. Попробуйте ещё раз.")
        logging.error(f"Ошибка: {e}")


@bot.message_handler(content_types=['document'])
def handle_image_document(message):
    stop_typing = threading.Event()
    typing_thread = threading.Thread(target=show_typing, args=(bot, message.chat.id, stop_typing))
    typing_thread.start()

    try:
        file_name = message.document.file_name.lower()
        if not file_name.endswith((".jpg", ".jpeg", ".png")):
            stop_typing.set()
            typing_thread.join()
            bot.send_message(message.chat.id,
                             "⚠️ Пожалуйста, отправьте изображение (JPG, PNG).")
            return

        file_info = bot.get_file(message.document.file_id)
        downloaded = bot.download_file(file_info.file_path)
        image = Image.open(io.BytesIO(downloaded)).convert("RGB")

        raw_text, corrected_text, errors = send_image_to_pipeline(image)

        user_data[message.chat.id] = {
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "errors": errors
        }

        stop_typing.set()
        typing_thread.join()

        bot.send_message(message.chat.id,
                         "✅ Документ успешно обработан.\nЧто вы хотите сделать дальше?",
                         reply_markup=text_action_keyboard(),
                         parse_mode="Markdown")

    except Exception as e:
        stop_typing.set()
        typing_thread.join()
        bot.send_message(message.chat.id,
                         "❌ Ошибка при обработке изображения-документа.")
        logging.error(f"[ERROR] Ошибка: {e}")


logging.info("Бот запущен. Ожидаю изображения...")
bot.polling(none_stop=True)

