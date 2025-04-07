import logging
import re
import torch
import csv
import os
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from telegram import ReplyKeyboardMarkup


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
TOKEN = ("Paste token")

LOG_DIR = "dir to save logs"
MODEL_PATH = "path where you storage your Transformer model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(LOG_DIR, exist_ok=True)


def clean_tweet(text):
    text = re.sub(r'http\S+', '', text)  # Удаление ссылок
    text = re.sub(r'@\w+', '', text)  # Удаление упоминаний
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9\s.,:;?!\-\']', '', text)
    text = text.strip()
    return text


def log_request(user_data: dict):
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(LOG_DIR, f"requests_{today}.csv")

    user_data.setdefault("username", "")
    user_data.setdefault("first_name", "")

    fieldnames = ["timestamp", "user_id", "username", "first_name", "text", "prediction"]

    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # Проверяем, пуст ли файл
            if os.stat(log_file).st_size == 0:
                writer.writeheader()
            writer.writerow(user_data)
        logger.info(f"Запись успешно добавлена: {user_data}")
    except Exception as e:
        logger.error(f"ОШИБКА ЗАПИСИ: {str(e)}", exc_info=True)


try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_safetensors=True,
        ignore_mismatched_sizes=True
    ).to(DEVICE)
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    exit(1)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    text = update.message.text
    timestamp = datetime.now().isoformat()
    prediction = None

    # Собираем базовые данные пользователя
    user_info = {
        "user_id": user.id,
        "username": user.username or "",  # Обработка None
        "first_name": user.first_name or "",  # Обработка None
    }

    try:
        inputs = tokenizer(
            clean_tweet(text),
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits).item()

        response = "⛔ Disaster detected!" if prediction == 1 else "✅ No disaster"

    except Exception as e:
        logger.error(f"Ошибка обработки: {str(e)}", exc_info=True)
        response = "⚠ Произошла ошибка"
        prediction = "Error"

    finally:
        # Формируем полные данные для лога
        log_data = {
            "timestamp": timestamp,
            **user_info,
            "text": text,
            "prediction": "Disaster" if prediction == 1 else "No disaster" if prediction == 0 else "Error"
        }

        try:
            log_request(log_data)
        except Exception as e:
            logger.error(f"КРИТИЧЕСКАЯ ОШИБКА ЛОГИРОВАНИЯ: {str(e)}")

        await update.message.reply_text(response)


# Остальные функции без изменений
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [["/help"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "Привет! Я анализирую текст на наличие информации о катастрофах.",
        reply_markup=reply_markup
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [["/help"]]
    await update.message.reply_text(
        "📋 Просто отправьте текст на английском языке для анализа",
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    )


def main() -> None:
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()


if __name__ == "__main__":
    print(f"|{TOKEN}|")
    main()