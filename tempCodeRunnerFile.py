import telebot
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import io
import os

bot = telebot.TeleBot('7654203891:AAFEb7yBUe5YqoP4ADJnl8Ipa7GzJlJjvt4')

processor_dir = os.path.join(".", "trocr_processor_v2")
model_dir = os.path.join(".", "trocr_model_v2")

processor = TrOCRProcessor.from_pretrained(processor_dir)
model = VisionEncoderDecoderModel.from_pretrained(model_dir)

@bot.message_handler(content_types=['text'])
def Get_Text_Message(message):
    print("Здравствуй :)")
    if message.text == "/start":
        bot.send_message(message.from_user.id,
                        "Привет, я бот для распознавания рукописного текста."
                        " Пришли мне фото, и я помогу тебе распознать, что на нем написано!")
    elif message.text == "/help":
        bot.send_message(message.from_user.id, "Напиши /start!")
    else:
        bot.send_message(message.from_user.id, "Прости... Я не понимаю тебя :(\nТы можешь написать /help, чтобы узнать больше!")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        image = Image.open(io.BytesIO(downloaded_file)).convert("RGB")

        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        bot.send_message(message.chat.id, f"📝 Распознанный текст:\n{text}")

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Произошла ошибка при распознавании изображения: {e}")
        print(e)


print("Бот запущен...")
bot.polling(none_stop=True)