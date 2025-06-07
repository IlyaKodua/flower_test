import requests
import os
import random
from typing import List
import base64

def get_image_paths(dataset_path: str, num_images: int = 5) -> List[str]:
    """
    Собирает случайные пути к изображениям из датасета.
    
    Args:
        dataset_path (str): Путь к корневой папке датасета.
        num_images (int): Количество изображений для выбора.
    
    Returns:
        List[str]: Список путей к изображениям.
    """
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    return random.sample(image_paths, min(num_images, len(image_paths)))

def send_image_in_base64(image_path: str, url: str):
    """
    Отправляет изображение в формате Base64 через POST-запрос.
    
    Args:
        image_path: Путь к изображению.
        url: URL для отправки запроса.
    
    Returns:
        dict: Ответ сервера.
    """
    try:
        # Читаем изображение и кодируем в Base64
        with open(image_path, "rb") as f:
            image_data = f.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Формируем JSON с Base64-строкой
        payload = {"image": base64_image}
        
        # Отправляем POST-запрос
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Проверяем, успешен ли запрос
        
        return response.json()
    
    except Exception as e:
        print(f"Ошибка при отправке изображения: {e}")
        return None
    
def test_search_endpoint(image_paths: List[str], url: str):
    """
    Тестирует API-эндпоинт, отправляя изображения и выводя результаты.
    
    Args:
        image_paths (List[str]): Список путей к тестовым изображениям.
        url (str): URL эндпоинта API.
    """
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Ошибка: Изображение {image_path} не найдено")
            continue
        
        # Отправляем POST-запрос с изображением
        try:
            resulr = send_image_in_base64(image_path, url)
            return resulr
        
        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {str(e)}")


dataset_path = "dataset"  # Путь к датасету
num_test_images = 5  # Количество тестовых изображений
api_url = "http://localhost:8080/search"  # URL API

# Собираем случайные изображения
image_paths = get_image_paths(dataset_path, num_test_images)



result = test_search_endpoint(image_paths, api_url)

print(result)