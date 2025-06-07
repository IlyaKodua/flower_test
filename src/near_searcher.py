import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Tuple
import pickle


class EmbeddingGetter:
    def __init__(self, path_to_weights : str, device):
        self.device = device
        self.embedding_model = models.efficientnet_v2_s()
        self.embedding_model.classifier[-1] = nn.Identity()  
        
        self.embedding_model.load_state_dict(torch.load(path_to_weights))
        self.embedding_model.to(device)
        self.embedding_model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
            
    def get_embedding(self, image):
        tensor = self.transform(image)
        with torch.no_grad():
            emb = self.embedding_model(tensor.unsqueeze(0).to(self.device)).detach().cpu().numpy()
        return emb[0]
    

class NearestFinder:

    def __init__(self, embeddings_dict_path, path_to_weights, device):

        self.embedding_getter = EmbeddingGetter(path_to_weights, device)
        with open(embeddings_dict_path, 'rb') as f:
            self.embeddings_dict = pickle.load(f)

    def find_nearest(self, image):
        query_embedding = self.embedding_getter.get_embedding(image)
        distances_with_images = self.find_top_k_similar(query_embedding)
        return distances_with_images

    @staticmethod
    def cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Вычисляет косинусное расстояние между двумя эмбеддингами.
        
        Args:
            emb1: Первый эмбеддинг (numpy массив или PyTorch тензор).
            emb2: Второй эмбеддинг (numpy массив или PyTorch тензор).
        
        Returns:
            float: Косинусное расстояние (1 - косинусное сходство).
        """
        # Нормализация
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        
        # Косинусное сходство
        cosine_similarity = np.dot(emb1, emb2)
        
        # Косинусное расстояние
        return 1.0 - cosine_similarity

    def find_top_k_similar(self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Находит топ-k ближайших изображений по косинусному расстоянию.
        
        Args:
            query_embedding: Эмбеддинг запроса.
            embeddings_dict: Словарь {путь_к_файлу: эмбеддинг}.
            k: Количество возвращаемых результатов.
        
        Returns:
            List[Tuple[str, float]]: Список из k пар (путь_к_файлу, косинусное_расстояние).
        """
        distances = []
        for img_path, emb in self.embeddings_dict.items():
            distance = self.cosine_distance(query_embedding, emb)
            distances.append((img_path, distance))
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]


