import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import hdbscan
import numpy as np
import torch
import umap.umap_ as umap
from numpy import ndarray
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.distance import euclidean
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class EmbeddingConfig:
    random_seed: int
    model_name: str


class Embedding:
    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg
        random.seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        torch.cuda.manual_seed_all(cfg.random_seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = SentenceTransformer(cfg.model_name)

    def to_embeddings(self, sentences: List[str]) -> ndarray:
        return self.model.encode(sentences)

    def cosine_similarity_(self, sentences: List[str]) -> List:
        embeddings = self.model.encode(sentences)
        return cosine_similarity(embeddings)

    def euclidean_distance(self, sentences: List[str]):
        embeddings = self.model.encode(sentences)
        dist_matrix = np.zeros((len(sentences), len(sentences)))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                dist_matrix[i][j] = euclidean(embeddings[i], embeddings[j])
        return dist_matrix


@dataclass
class ClusteringConfig:
    use_umap: bool
    random_seed: int
    umap_params: dict
    hdbscan_params: dict


class Clustering:
    def __init__(self, cfg: ClusteringConfig) -> None:
        self.cfg = cfg
        if cfg.use_umap:
            self.reducer = umap.UMAP(
                n_neighbors=cfg.umap_params.n_neighbors,
                min_dist=cfg.umap_params.min_dist,
                n_components=cfg.umap_params.n_components,
                metric=cfg.umap_params.metric,
                random_state=cfg.random_seed
            )
        else:
            raise NotImplementedError("Currently, only UMAP is supported for clustering.")

        self.cluster = hdbscan.HDBSCAN(
            min_cluster_size=cfg.hdbscan_params.min_cluster_size,
            min_samples=cfg.hdbscan_params.min_samples,
            metric=cfg.hdbscan_params.metric,
            cluster_selection_method=cfg.hdbscan_params.cluster_selection_method,
            allow_single_cluster=cfg.hdbscan_params.allow_single_cluster,
            prediction_data=True
        )
    def run_clustering(self, embeddings: ndarray):
        reduced_embeddings_for_clustering = self.reducer.fit_transform(embeddings)
        input_for_hdbscan = reduced_embeddings_for_clustering

        cluster_labels = self.cluster.fit_predict(input_for_hdbscan)

        return cluster_labels

    def predict_cluster(self, new_embedding: np.ndarray) -> tuple[int, float]:
        if len(new_embedding.shape) == 1:
            new_embedding = new_embedding.reshape(1, -1)

        reduced_embedding = self.reducer.transform(new_embedding)

        label, strengths = hdbscan.approximate_predict(self.cluster, reduced_embedding)
        predicted_label = label[0]

        if predicted_label == -1:
            strength = 0.0
        else:
            try:
                strength = strengths[0]
            except IndexError:
                strength = 0.0

        return predicted_label, strength


class KnowRouter:
    def __init__(self, cfg) -> None:
        cfg = OmegaConf.create(cfg) if not isinstance(cfg, DictConfig) else cfg

        self.cfg = cfg
        self.embedding = Embedding(cfg.embedding)
        print('Embedding model:', cfg.embedding.model_name)
        self.clustering = Clustering(cfg.clustering)

        self.anchors = []
        self.route_table = None
        self.built = False

        if self.cfg.two_stages:
            self.boundary_embedding = Embedding(
                EmbeddingConfig(
                    random_seed=cfg.embedding.random_seed,
                    model_name=cfg.boundary_model_name
                )
            )
            print(f"Boundary model: {cfg.boundary_model_name}")
        else:
            print('Two stages routing is not enabled.')

    def build_route_table(self, prompt_list: List[str]) -> None:
        embeddings = self.embedding.to_embeddings(prompt_list)
        cluster_labels = self.clustering.run_clustering(embeddings)

        if self.cfg.two_stages:
            self.anchors = prompt_list
            self.anchor_embeddings = self.boundary_embedding.to_embeddings(self.anchors)

        self.route_table = {
            prompt: cluster_id
            for prompt, cluster_id in zip(prompt_list, cluster_labels)
        }
        self.built = True

    def boundary_route(self, prompt: str, threshold: float) -> bool:
        prompt_embedding = self.boundary_embedding.to_embeddings([prompt])

        similarities = cosine_similarity(prompt_embedding, self.anchor_embeddings)

        max_similarity = np.max(similarities[0])

        should_route_to_original = max_similarity < threshold

        return should_route_to_original

    def route(self, prompt: str) -> int:
        if not self.cfg.use_multi_ffn:
            if self.boundary_route(prompt, self.cfg.boundary_threshold):
                return -2
            else:
                return -1

        if not self.built:
            raise RuntimeError("Router not built. Call build_route_table() first.")
        if prompt in self.route_table:
            cluster_id = self.route_table[prompt]
            print(f'[Domain Router] X_input => {cluster_id}')
            return self.route_table[prompt]


        if self.cfg.two_stages and self.boundary_route(prompt, self.cfg.boundary_threshold):
            return -2
        embedding = self.embedding.to_embeddings([prompt])[0]
        cluster_id, _ = self.clustering.predict_cluster(embedding)
        print(f'[Domain Router] X_input => {cluster_id}')
        return cluster_id

    def route_with_confidence(self, prompt: str) -> tuple[int, float]:
        if not self.built:
            raise RuntimeError("Router not built. Call build_route_table() first.")

        embedding = self.embedding.to_embeddings([prompt])[0]
        cluster_id, confidence = self.clustering.predict_cluster(embedding)
        return cluster_id, confidence


    def _count_similarity(self):
        pass

    def get_num_clusters(self) -> int:
        if not self.built:
            raise RuntimeError("not yet cluster. build_route_table()")

        if not self.cfg.use_multi_ffn:
            return 1
        labels = self.clustering.cluster.labels_

        unique_labels = set(labels)
        num_clusters = len(unique_labels - {-1})

        return num_clusters

    def get_num_outlier(self) -> int:
        if not self.cfg.use_multi_ffn:
            return 0

        if not self.built:
            raise RuntimeError("not yet cluster. build_route_table()")
        labels = self.clustering.cluster.labels_

        num_outliers = 0
        for i in labels:
            if i == -1:
                num_outliers += 1

        return num_outliers

    def save(self, save_dir: str) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "router.pkl", "wb") as f:
            pickle.dump({
                "route_table": self.route_table,
                "built": self.built,
                "cfg": self.cfg
            }, f)

        self.embedding.model.save(str(save_path / "embedding_model"))

        with open(save_path / "clustering.pkl", "wb") as f:
            pickle.dump({
                "reducer": self.clustering.reducer,
                "cluster": self.clustering.cluster
            }, f)

    @classmethod
    def load(cls, save_dir: str) -> "KnowRouter":
        save_path = Path(save_dir)

        with open(save_path / "router.pkl", "rb") as f:
            router_data = pickle.load(f)

        router = cls(router_data["cfg"])
        router.route_table = router_data["route_table"]
        router.built = router_data["built"]

        router.embedding.model = SentenceTransformer(str(save_path / "embedding_model"))

        with open(save_path / "clustering.pkl", "rb") as f:
            clustering_data = pickle.load(f)
        router.clustering.reducer = clustering_data["reducer"]
        router.clustering.cluster = clustering_data["cluster"]

        return router


class ScopeRouter:
    def __init__(self, cfg) -> None:
        cfg = OmegaConf.create(cfg) if not isinstance(cfg, DictConfig) else cfg
        self.cfg = cfg



if __name__ == '__main__':
    pass
