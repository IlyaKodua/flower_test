from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
import yaml
from near_searcher import NearestFinder

app = FastAPI()

with open("configs/config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
    
# Загружаем модель и словарь эмбеддингов
device = torch.device(config["device"])
finder_sim_images = NearestFinder(config["embeddings_path"], 
                                  config["weights_path"],
                                  device)

@app.post("/search")
async def search_image(file: UploadFile = File(...)):
    """
    Принимает изображение и возвращает топ-5 ближайших изображений из тестовой выборки.
    
    Args:
        file: Загруженное изображение.
    
    Returns:
        dict: JSON с топ-5 путями и расстояниями.
    """
    # Проверяем тип файла
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    # Читаем изображение
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # Находим топ-5 ближайших изображений
    top_k_results = finder_sim_images.find_nearest(image)

    # Формируем ответ
    result = [
        {"path": img_path, "simularity": float(1 - distance)}
        for img_path, distance in top_k_results
    ]
    return {"results": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)