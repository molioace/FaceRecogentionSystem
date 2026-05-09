import faiss
import numpy as np

from app.config import FAISS_INDEX_PATH


class FaceIndex:
    def __init__(self, embedding_dim=512, threshold=0.6):
        self.embedding_dim = embedding_dim
        self.threshold = threshold
        self.users = []

        try:
            if FAISS_INDEX_PATH.exists() and FAISS_INDEX_PATH.stat().st_size > 0:
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            else:
                self.index = faiss.IndexFlatIP(embedding_dim)
        except Exception:
            self.index = faiss.IndexFlatIP(embedding_dim)

    def save(self):
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))

    def add_face(self, user_id, embedding):
        embedding = embedding.astype(np.float32).reshape(1, -1)

        self.index.add(embedding)
        self.users.append(user_id)

        self.save()

    def search(self, embedding):
        if self.index.ntotal == 0:
            return {
                "identity": "Unknown",
                "similarity": 0.0,
                "matched": False
            }

        embedding = embedding.astype(np.float32).reshape(1, -1)

        scores, indexes = self.index.search(embedding, 1)

        similarity = float(scores[0][0])
        index_id = int(indexes[0][0])

        if index_id < 0 or index_id >= len(self.users):
            return {
                "identity": "Unknown",
                "similarity": similarity,
                "matched": False
            }

        if similarity >= self.threshold:
            return {
                "identity": self.users[index_id],
                "similarity": similarity,
                "matched": True
            }

        return {
            "identity": "Unknown",
            "similarity": similarity,
            "matched": False
        }

    def save(self):
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))