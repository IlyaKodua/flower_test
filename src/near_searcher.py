import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from typing import List, Tuple
import pickle

class EmbeddingGetter:
    """A class to extract image embeddings using a pre-trained EfficientNet model.

    This class initializes a pre-trained EfficientNetV2-S model, modifies its classifier
    to output embeddings, and applies necessary image transformations for embedding extraction.

    Attributes:
        device (torch.device): The device (CPU/GPU) to run the model on.
        embedding_model (nn.Module): The modified EfficientNetV2-S model.
        transform (transforms.Compose): Image preprocessing transformations.
    """

    def __init__(self, path_to_weights: str, device: torch.device):
        """Initialize the EmbeddingGetter with model weights and device.

        Args:
            path_to_weights (str): Path to the pre-trained model weights file.
            device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').

        Example:
            >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            >>> embedder = EmbeddingGetter('weights.pth', device)
        """
        self.device = device
        self.embedding_model = models.efficientnet_v2_s()
        self.embedding_model.classifier[-1] = nn.Identity()
        
        self.embedding_model.load_state_dict(torch.load(path_to_weights,
                                                        map_location=torch.device('cpu')))
        self.embedding_model.to(device)
        self.embedding_model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract the embedding for a given image.

        Args:
            image (PIL.Image.Image): Input image to process.

        Returns:
            np.ndarray: 1D numpy array containing the image embedding.

        Example:
            >>> from PIL import Image
            >>> img = Image.open('example.jpg')
            >>> embedding = embedder.get_embedding(img)
            >>> embedding.shape
            (1280,)
        """
        tensor = self.transform(image)
        with torch.no_grad():
            emb = self.embedding_model(tensor.unsqueeze(0).to(self.device)).detach().cpu().numpy()
        return emb[0]


class NearestFinder:
    """A class to find the nearest images based on cosine distance between embeddings.

    This class uses an EmbeddingGetter to extract embeddings and compares them against
    a pre-computed dictionary of embeddings to find the most similar images.

    Attributes:
        embedding_getter (EmbeddingGetter): Instance to extract image embeddings.
        embeddings_dict (Dict[str, np.ndarray]): Dictionary mapping image paths to embeddings.
    """

    def __init__(self, embeddings_dict_path: str, path_to_weights: str, device: torch.device):
        """Initialize the NearestFinder with embeddings dictionary and model weights.

        Args:
            embeddings_dict_path (str): Path to the pickled embeddings dictionary.
            path_to_weights (str): Path to the pre-trained model weights file.
            device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').

        Example:
            >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            >>> finder = NearestFinder('embeddings.pkl', 'weights.pth', device)
        """
        self.embedding_getter = EmbeddingGetter(path_to_weights, device)
        with open(embeddings_dict_path, 'rb') as f:
            self.embeddings_dict = pickle.load(f)

    def find_nearest(self, image: Image.Image) -> List[Tuple[str, float]]:
        """Find the nearest images to the input image based on embedding similarity.

        Args:
            image (PIL.Image.Image): Input image to find nearest matches for.

        Returns:
            List[Tuple[str, float]]: List of tuples containing image paths and their
                                    cosine distances to the input image.

        Example:
            >>> from PIL import Image
            >>> img = Image.open('query.jpg')
            >>> results = finder.find_nearest(img)
            >>> for path, distance in results:
            ...     print(f"Image: {path}, Distance: {distance}")
        """
        query_embedding = self.embedding_getter.get_embedding(image)
        distances_with_images = self.find_top_k_similar(query_embedding)
        return distances_with_images

    @staticmethod
    def cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine distance between two embeddings.

        The cosine distance is defined as 1 minus the cosine similarity, where
        cosine similarity is the dot product of normalized embeddings.

        Args:
            emb1 (np.ndarray): First embedding (numpy array).
            emb2 (np.ndarray): Second embedding (numpy array).

        Returns:
            float: Cosine distance (1 - cosine similarity).

        Example:
            >>> emb1 = np.array([1.0, 0.0])
            >>> emb2 = np.array([0.0, 1.0])
            >>> distance = NearestFinder.cosine_distance(emb1, emb2)
            >>> print(distance)  # Expected: 1.0
        """
        # Normalize embeddings
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        
        # Calculate cosine similarity
        cosine_similarity = np.dot(emb1, emb2)
        
        # Return cosine distance
        return 1.0 - cosine_similarity

    def find_top_k_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Find top-k nearest images based on cosine distance.

        Args:
            query_embedding (np.ndarray): Embedding of the query image.
            k (int, optional): Number of nearest images to return. Defaults to 5.

        Returns:
            List[Tuple[str, float]]: List of k tuples containing image paths and
                                    their cosine distances, sorted by distance.

        Example:
            >>> query_emb = np.random.rand(1280)
            >>> results = finder.find_top_k_similar(query_emb, k=3)
            >>> len(results)
            3
        """
        distances = []
        for img_path, emb in self.embeddings_dict.items():
            distance = self.cosine_distance(query_embedding, emb)
            distances.append((img_path, distance))
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]