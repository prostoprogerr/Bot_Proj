from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
from recognizer_service.pipeline import process_image_pipeline

app = FastAPI()

@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    recognized_text, corrected_text, errors = process_image_pipeline(image)

    return {
        "recognized_text": recognized_text,
        "corrected_text": corrected_text,
        "errors": errors
    }
