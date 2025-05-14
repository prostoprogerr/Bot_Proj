import telebot
import io
import logging
from recognizer import process_image_pipeline
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

bot = telebot.TeleBot('7654203891:AAFEb7yBUe5YqoP4ADJnl8Ipa7GzJlJjvt4')


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
        bot.send_message(message.chat.id, "📥 Получено изображение-документ. \n🧠 Распознаю текст...")

        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded = bot.download_file(file_info.file_path)
        image = Image.open(io.BytesIO(downloaded)).convert("RGB")

        raw_text, corrected_text, errors = process_image_pipeline(image)

        bot.send_message(message.chat.id, f"📝 *Распознанный текст:*\n`{raw_text}", parse_mode="Markdown")
        bot.send_message(message.chat.id, f"✅ *Исправленный текст:*\n`{corrected_text}", parse_mode="Markdown")

        if errors:
            bot.send_message(message.chat.id, f"🔍 *Обнаружены ошибки:*\n{errors}", parse_mode="HTML")
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
            bot.send_message(message.chat.id, f"🔍 *Обнаружены ошибки:*\n{errors}", parse_mode="HTML")
        else:
            bot.send_message(message.chat.id, "🎉 Ошибок не найдено!")

    except Exception as e:
        bot.send_message(message.chat.id, "❌ Произошла ошибка при обработке изображения-документа.")
        print(f"[ERROR] Ошибка с изображением-документом: {e}")


print("🚀 Бот запущен. Ожидаю изображения...")
bot.polling(none_stop=True)
