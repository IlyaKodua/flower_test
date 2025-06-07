import requests
import os
import random
from typing import List

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

def test_search_endpoint(image_paths: List[str], url: str = "http://localhost:8080/search"):
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
            with open(image_path, "rb") as f:
                files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
                response = requests.post(url, files=files)
            
            # Проверяем статус ответа
            if response.status_code != 200:
                print(f"Ошибка для {image_path}: {response.status_code}, {response.json().get('detail', 'Нет деталей')}")
                continue
            
            # Получаем результаты
            results = response.json()
            return results
        
        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {str(e)}")


dataset_path = "dataset"  # Путь к датасету
num_test_images = 5  # Количество тестовых изображений
api_url = "http://localhost:8080/search"  # URL API

# Собираем случайные изображения
image_paths = get_image_paths(dataset_path, num_test_images)



result = test_search_endpoint(image_paths, api_url)

print(result)