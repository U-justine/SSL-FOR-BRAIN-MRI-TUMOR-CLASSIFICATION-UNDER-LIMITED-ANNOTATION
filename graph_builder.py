
from typing import Optional, List, Tuple, Set
import warnings
import torch
import torch.nn.functional as F


class GraphBuilder:

    def __init__(self, config):
        self.threshold = config.similarity_threshold
        self.use_coref = config.use_coreference_edges

        if self.use_coref:
            warnings.warn(
                "Coreference edges are enabled but not yet implemented. "
                "Falling back to similarity-only graph construction.",
                UserWarning,
            )

    def compute_cosine_similarity(
        self, embeddings: torch.Tensor
    ) -> torch.Tensor:
        if embeddings.dim() != 2:
            raise ValueError(
                f"Expected 2D embeddings (N, D), got {embeddings.dim()}D "
                f"with shape {embeddings.shape}"
            )

        # F.normalize handles zero vectors gracefully (returns zero)
        normalized = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.mm(normalized, normalized.t())

        # Clamp to [0, 1] — negative similarities are not meaningful
        # for discourse graph connectivity
        sim_matrix = sim_matrix.clamp(min=0.0, max=1.0)

        return sim_matrix

    def _build_coreference_edges(
        self, sentences: List[str], n: int, device: torch.device
    ) -> torch.Tensor:
        # TODO: Integrate spaCy/neuralcoref or AllenNLP coreference resolver
        # For now, return empty adjacency (no additional edges)
        return torch.zeros(n, n, device=device)

    def build_graph(
        self,
        sentence_embeddings: torch.Tensor,
        sentences: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sentence_embeddings.dim() != 2:
            raise ValueError(
                f"Expected 2D sentence_embeddings (N, D), got "
                f"{sentence_embeddings.dim()}D with shape {sentence_embeddings.shape}"
            )

        n = sentence_embeddings.size(0)
        device = sentence_embeddings.device

        # ── Handle trivial cases ──
        if n == 0:
            empty = torch.zeros(0, 0, device=device)
            return sentence_embeddings, empty, empty

        if n == 1:
            ones = torch.ones(1, 1, device=device)
            return sentence_embeddings, ones, ones

        # ── Validate sentences if provided ──
        if sentences is not None and len(sentences) != n:
            raise ValueError(
                f"Number of sentences ({len(sentences)}) does not match "
                f"number of embeddings ({n})"
            )

        # ── Cosine similarity ──
        sim_matrix = self.compute_cosine_similarity(sentence_embeddings)

        # ── Threshold edges ──
        edge_weights = torch.where(
            sim_matrix > self.threshold,
            sim_matrix,
            torch.zeros_like(sim_matrix),
        )

        # ── Self-loops (always present) ──
        edge_weights.fill_diagonal_(1.0)

        # ── Coreference edges (optional) ──
        if self.use_coref and sentences is not None:
            coref_adj = self._build_coreference_edges(sentences, n, device)
            # Union: add coreference edges with default weight of threshold + ε
            # so they survive thresholding
            coref_weight = self.threshold + 0.1
            coref_mask = (coref_adj > 0) & (edge_weights == 0)
            edge_weights = edge_weights + coref_mask.float() * coref_weight

        # ── Binary adjacency ──
        adj_matrix = (edge_weights > 0).float()

        # ── Connectivity diagnostics ──
        degree = adj_matrix.sum(dim=1) - 1  # Subtract self-loop
        isolated_count = (degree == 0).sum().item()
        if isolated_count > 0 and n > 1:
            warnings.warn(
                f"{isolated_count}/{n} sentences are isolated (no neighbors). "
                f"Consider lowering similarity_threshold (currently {self.threshold:.2f}).",
                UserWarning,
            )

        return sentence_embeddings, adj_matrix, edge_weights

    def get_graph_stats(
        self, adj_matrix: torch.Tensor
    ) -> dict:
        n = adj_matrix.size(0)
        if n == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "density": 0.0,
                "avg_degree": 0.0,
                "isolated_nodes": 0,
            }

        # Exclude self-loops for edge/degree counting
        adj_no_self = adj_matrix.clone()
        adj_no_self.fill_diagonal_(0)

        num_edges_directed = adj_no_self.sum().item()
        # Undirected: divide by 2 (symmetric matrix)
        num_edges = int(num_edges_directed / 2)

        degree = adj_no_self.sum(dim=1)
        max_possible_edges = n * (n - 1) / 2

        return {
            "num_nodes": n,
            "num_edges": num_edges,
            "density": num_edges / max_possible_edges if max_possible_edges > 0 else 0.0,
            "avg_degree": degree.mean().item(),
            "max_degree": degree.max().item(),
            "min_degree": degree.min().item(),
            "isolated_nodes": (degree == 0).sum().item(),
        }