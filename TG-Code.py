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
    if message.text == "/start":
        bot.send_message(message.from_user.id,
                        "Привет еще раз! Я здесь, чтобы помочь тебе распознать русский рукописный текст. Просто отправь фото с текстом, и я постараюсь его обработать."
                        "\nЕсли что-то пойдет не так, не переживай — я еще учусь! "
                        "\n😊Напоминаю, поддерживаются форматы: jpeg, jpg, png, bmp.")
    elif message.text == "/help":
        bot.send_message(message.from_user.id, "Вот что я умею:"
                                               "\nРаспознавать русский рукописный текст с изображений.  Работать с файлами в форматах jpeg, jpg, png, bmp."
                                               "\nСоветы:"
                                               "\nДля лучшего распознавания старайтесь делать четкие фотографии текста. Пока я лучше всего справляюсь с одним словом. Если текст длинный, результат может быть менее точным. "
                                               "\n⚠️Если что-то пошло не так, напишите моему разработчику, чтобы он мог улучшить меня. Спасибо за понимание!")
    else:
        bot.send_message(message.from_user.id, "Прости... Я не понимаю тебя :(\nТы можешь написать /help, чтобы узнать больше!")

@bot.message_handler(content_types=['photo'])
def Handle_Photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        image = Image.open(io.BytesIO(downloaded_file)).convert("RGB")

        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        bot.send_message(message.chat.id, f"📝✅Готово! Если результат не совсем точный, попробуйте отправить другое изображение или улучшить качество фото. Вот какой текст я распознал:\n{text}")

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Произошла ошибка при распознавании изображения: {e}")
        print(e)


print("Привет! 👋 Я — бот для распознавания русского рукописного текста.\nПросто отправь мне изображение с текстом (в формате jpeg, jpg, png или bmp), и я постараюсь его распознать.\n❗️ Важно : на данный момент я лучше всего работаю с одним словом. Если на изображении несколько слов, качество распознавания может быть ниже."
      "\nДля дополнительной информации используй команды:"
      "\n/start — начать работу со мной/help — получить справку"
      "\nЖду ваше фото! 📸📝")
bot.polling(none_stop=True)
