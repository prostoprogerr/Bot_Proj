import requests, logging

def check_yandex_spelling(text):
    """Проверяет орфографию с помощью Яндекс-спеллера."""
    try:
        response = requests.get(
            "https://speller.yandex.net/services/spellservice.json/checkText",
            params={"text": text, "lang": "ru"}
        )
        results = response.json()
        corrected_text = text
        log = ""

        for item in reversed(results):
            if item["s"]:
                suggestion = item["s"][0]
                start = item["pos"]
                end = start + item["len"]
                corrected_text = corrected_text[:start] + suggestion + corrected_text[end:]
                log += f"• <code>{item['word']}</code> → <b>{suggestion}</b>\n"

        return corrected_text, log

    except Exception as e:
        logging.info(f"[ERROR] Yandex Speller error: {e}")
        return text, "⚠️ <i>Не удалось проверить орфографию через Яндекс.</i>\n"


def check_grammar_tool(text):
    """Проверяет грамматику и стиль с помощью LanguageTool."""
    import language_tool_python
    try:
        tool = language_tool_python.LanguageTool('ru-RU')
        matches = tool.check(text)
        corrected_text = language_tool_python.utils.correct(text, matches)

        log = ""
        for match in matches:
            context = match.context.replace('\n', ' ')
            log += f"• <b>{match.message}</b>\n  ⤷ <code>{context.strip()}</code>\n"

        tool.close()
        return corrected_text, log

    except Exception as e:
        logging.info(f"[ERROR] LanguageTool error: {e}")
        return text, "⚠️ <i>Не удалось проверить грамматику через LanguageTool.</i>"

def check_spelling_and_grammar(text):
    """Использует Яндекс-спеллер и LanguageTool для проверки орфографии и грамматики."""
    yandex_corrected, yandex_log = check_yandex_spelling(text)
    final_text, lt_log = check_grammar_tool(yandex_corrected)

    full_log = ""
    if yandex_log.strip():
        full_log += "<b>🧹 Орфография:</b>\n" + yandex_log + "\n"
    if lt_log.strip():
        full_log += "<b>🔍 Грамматика и стиль:</b>\n" + lt_log

    return final_text.strip(), full_log.strip()
