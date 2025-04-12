import telebot;

bot = telebot.TeleBot('7654203891:AAFEb7yBUe5YqoP4ADJnl8Ipa7GzJlJjvt4');

@bot.message_handler(content_types=['text'])

def Get_Text_Message(message):
    print("Здравтсвуй :)")
    if message.text == "/start":
        bot.send_message(message.from_user.id, "Привет, я телеграмм-бот для перевода слов с английского языка на русский. Пришли мне фото и я помогу тебе перевести то, что на нем написано!")
    elif message.text == "/help":
        bot.send_message(message.from_user.id, "Напиши /start!")
    else:
        bot.send_message(message.from_user.id, "Прости... Я не понимаю тебя :(\nТы можешь написать /help, чтобы узнать больше!")

bot.polling(none_stop=True, interval=0)