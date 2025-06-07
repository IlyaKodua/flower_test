from fastapi import FastAPI, HTTPException
from PIL import Image
import io
import torch
import yaml
from near_searcher import NearestFinder
import base64
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

app = FastAPI()

with open("configs/config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
    
# Загружаем модель и словарь эмбеддингов
device = torch.device(config["device"])
finder_sim_images = NearestFinder(config["embeddings_path"], 
                                  config["weights_path"],
                                  device)

@app.post("/search")
async def search_image(data: dict):
    """
    Принимает изображение и возвращает топ-5 ближайших изображений из тестовой выборки.
    
    Args:
        data: Запрос
    
    Returns:
        dict: JSON с топ-5 путями и расстояниями.
    """
    logger.info(f"Start search")
    image_data = base64.b64decode(data["image"])
    try:
        logger.info(f"read image")
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
    except Exception:
        logger.error(f"read image error")
        raise HTTPException(status_code=400, detail="Невалидное изображение")

    logger.info(f"find nearest")
    # Находим топ-5 ближайших изображений
    top_k_results = finder_sim_images.find_nearest(image)

    logger.info(f"make result")
    # Формируем ответ
    result = [
        {"path": img_path, "simularity": float(1 - distance)}
        for img_path, distance in top_k_results
    ]
    return {"results": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)