# 🤖 Bot_Proj — Распознавание и коррекция рукописного текста

Этот проект представляет собой систему обработки изображений с рукописным русским текстом: она обнаруживает слова на изображении с помощью YOLOv5, обрезает их, распознаёт с помощью TrOCR, а затем проверяет орфографию и грамматику.

---

## 📁 Структура проекта

```Bot_Proj/
├── .venv/                          
├── bot.py                          # Основной скрипт 
├── recognizer.py                   # Основная логика обработки изображений и текста
├── requirements.txt                # Список зависимостей Python
├── LICENSE                         # Лицензия проекта
├── README.md                       # Документация проекта
├── models/                         # Папка с весами моделей
│   ├── yolov5/                     # Веса для YOLOv5 (best.pt)
│   └── trocr/
│       └── v3/
│           ├── model/              # Веса TrOCR (pytorch_model.bin, config.json и др.)
│           └── processor/          # Tokenizer и Feature Extractor
├── bot_utils/                      # Вспомогательные модули
│   ├── __init__.py                 # Инициализация пакета
│   ├── crop.py                     # Чтение, сортировка и обрезка bbox
│   ├── resize.py                   # Преобразование размера изображений
│   └── check_spelling.py           # Проверка орфографии и грамматики
└── .gitignore                      
```

## Используемое ПО

- [YOLOv5](https://huggingface.co/Ultralytics/YOLOv5) — модель обнаружения объектов (вызов через `subprocess`). Лицензия: **GPLv3**
- [TrOCR (base-handwritten)](https://huggingface.co/microsoft/trocr-base-handwritten) — модель от Microsoft для распознавания рукописного текста. Лицензия: **MIT**
- [Pillow](https://pillow.readthedocs.io/en/stable/index.html) — для обработки изображений
- [Transformers](https://huggingface.co/docs/transformers/index) — для загрузки TrOCR
- [Yandex Speller API](https://yandex.ru/dev/speller/) — проверка орфографии
- [LanguageTool](https://languagetool.org/) — проверка грамматики

## 📌 Особенности

- Сортировка слов по строкам с учётом центра и высоты bbox
- Коррекция текста через два API: Yandex Speller и LanguageTool
- Поддержка CUDA (если доступна)
- Вывод логов в консоль

## 🧾 Лицензия

См. [LICENSE](LICENSE)