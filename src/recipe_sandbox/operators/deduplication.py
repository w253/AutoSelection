from __future__ import annotations

from typing import List, Sequence

from recipe_sandbox.operators.base import DedupOperator
from recipe_sandbox.operators.helpers import cosine_similarity, resolve_path, sample_to_text
from recipe_sandbox.schema.types import CanonicalSample
from recipe_sandbox.utils.hashing import stable_md5


class SemanticDedupOperator(DedupOperator):
    name = "semantic_dedup"

    def transform(self, dataset: Sequence[CanonicalSample]) -> List[CanonicalSample]:
        strategy = self.config.get("strategy", "exact")
        if strategy == "embedding_cosine":
            return self._embedding_dedup(dataset)
        elif strategy == "minhash":
            return self._minhash_dedup(dataset)
        return self._exact_dedup(dataset)

    def _minhash_dedup(self, dataset: Sequence[CanonicalSample]) -> List[CanonicalSample]:
        if not dataset:
            return []
            
        from datasketch import MinHash, MinHashLSH
        import os
        from concurrent.futures import ThreadPoolExecutor

        threshold = float(self.config.get("jaccard_threshold", 0.8))
        num_perm = int(self.config.get("num_perm", 128))
        ngram_size = int(self.config.get("ngram_size", 5))

        # Phase 1: Parallel MinHash computation (CPU-bound, independent per sample)
        def _compute_minhash(sample: CanonicalSample) -> MinHash:
            text = sample_to_text(sample).lower()
            tokens = text.split()
            if len(tokens) < ngram_size:
                ngrams = set([" ".join(tokens)]) if tokens else set()
            else:
                ngrams = set(
                    " ".join(tokens[j : j + ngram_size])
                    for j in range(len(tokens) - ngram_size + 1)
                )
            m = MinHash(num_perm=num_perm)
            for ngram in ngrams:
                m.update(ngram.encode("utf8"))
            return m

        num_workers = min(os.cpu_count() or 4, 16, len(dataset))
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            minhashes = list(pool.map(_compute_minhash, dataset))

        # Phase 2: Sequential LSH query + insert (must be serial for correctness)
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        outputs: List[CanonicalSample] = []
        duplicates = 0

        for i, (sample, m) in enumerate(zip(dataset, minhashes)):
            result = lsh.query(m)
            if result:
                duplicates += 1
                continue
            lsh.insert(f"doc_{i}", m)
            outputs.append(sample)

        self._trace.notes["dedup_strategy"] = "minhash"
        self._trace.notes["dedup_workers"] = num_workers
        self._trace.cost.extra["duplicates_removed"] = duplicates
        self._trace.cost.extra["jaccard_threshold"] = threshold
        self._trace.cost.extra["ngram_size"] = ngram_size
        return outputs

    def _exact_dedup(self, dataset: Sequence[CanonicalSample]) -> List[CanonicalSample]:
        seen = set()
        outputs: List[CanonicalSample] = []
        duplicates = 0

        for sample in dataset:
            fingerprint = stable_md5(sample_to_text(sample).lower().split())
            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)
            outputs.append(sample)

        self._trace.notes["dedup_strategy"] = "exact"
        self._trace.cost.extra["duplicates_removed"] = duplicates
        return outputs

    def _embedding_dedup(self, dataset: Sequence[CanonicalSample]) -> List[CanonicalSample]:
        embedding_path = self.config.get("embedding_path", "metadata.extra.embedding")
        keep_missing_embeddings = bool(self.config.get("keep_missing_embeddings", True))
        threshold = float(self.config.get("threshold", 0.98))

        kept_embeddings: List[List[float]] = []
        outputs: List[CanonicalSample] = []
        duplicates = 0

        for sample in dataset:
            embedding = resolve_path(sample, embedding_path)
            if embedding is None:
                if keep_missing_embeddings:
                    outputs.append(sample)
                continue

            vector = [float(value) for value in embedding]
            if any(cosine_similarity(vector, kept) >= threshold for kept in kept_embeddings):
                duplicates += 1
                continue

            kept_embeddings.append(vector)
            outputs.append(sample)

        self._trace.notes["dedup_strategy"] = "embedding_cosine"
        self._trace.cost.extra["duplicates_removed"] = duplicates
        self._trace.cost.extra["embedding_threshold"] = threshold
        return outputs


class SemDeDupOperator(DedupOperator):
    """SemDeDup: Semantic deduplication via SAE sparse feature clustering.

    Real implementation using cached ``sae_topk`` sparse features:
    1. Build sparse vectors from cached sae_topk for each sample.
    2. Cluster with mini-batch K-means.
    3. Within each cluster, remove samples whose cosine similarity
       to an already-kept sample exceeds ``cosine_threshold``.

    Paper: https://arxiv.org/abs/2303.09540 (ICLR 2023)
    Typical removal: 30-50% with threshold 0.5-0.7.

    Config keys:
        cosine_threshold: float (0.5-0.7) — intra-cluster dedup threshold
        num_clusters: int — K-means clusters (must be explicitly specified)
    """
    name = "semdedup"

    def transform(self, dataset: Sequence[CanonicalSample]) -> List[CanonicalSample]:
        import numpy as np
        from scipy import sparse as sp
        import logging

        log = logging.getLogger(__name__)

        if "cosine_threshold" not in self.config:
            raise ValueError(
                "SemDeDupOperator requires explicit 'cosine_threshold' parameter. "
                "Recipe must specify it (e.g. cosine_threshold: 0.7)."
            )
        if "num_clusters" not in self.config:
            raise ValueError(
                "SemDeDupOperator requires explicit 'num_clusters' parameter. "
                "Recipe must specify it (e.g. num_clusters: 500)."
            )
        cosine_threshold = float(self.config["cosine_threshold"])
        num_clusters = int(self.config["num_clusters"])

        from recipe_sandbox.scoring.sparse_features import get_sparse_features

        rows, cols, vals = [], [], []
        valid_indices = []
        d_sae = 0

        for i, sample in enumerate(dataset):
            topk = get_sparse_features(sample)
            if topk and topk.get("indices"):
                indices = topk["indices"]
                values = topk["values"]
                max_idx = max(indices)
                if max_idx >= d_sae:
                    d_sae = max_idx + 1
                row_id = len(valid_indices)
                rows.extend([row_id] * len(indices))
                cols.extend(indices)
                vals.extend(values)
                valid_indices.append(i)

        if not valid_indices or d_sae == 0:
            self._trace.notes["dedup_strategy"] = "semdedup_no_features"
            self._trace.cost.extra["duplicates_removed"] = 0
            return list(dataset)

        n_samples = len(valid_indices)
        log.info("semdedup: building sparse matrix %d × %d ...", n_samples, d_sae)

        matrix = sp.csr_matrix(
            (np.array(vals, dtype=np.float32), (rows, cols)),
            shape=(n_samples, d_sae),
        )

        # L2-normalize rows for cosine similarity
        from sklearn.preprocessing import normalize as sk_normalize
        matrix_normed = sk_normalize(matrix, norm='l2', copy=True)

        # K-means clustering (sklearn supports sparse input)
        actual_clusters = min(num_clusters, n_samples)
        log.info("semdedup: running MiniBatchKMeans (k=%d) ...", actual_clusters)
        if actual_clusters < 2:
            cluster_labels = np.zeros(n_samples, dtype=int)
        else:
            try:
                from sklearn.cluster import MiniBatchKMeans
                kmeans = MiniBatchKMeans(
                    n_clusters=actual_clusters,
                    random_state=42,
                    batch_size=min(1024, n_samples),
                    n_init=1,
                )
                cluster_labels = kmeans.fit_predict(matrix_normed)
            except ImportError:
                rng = np.random.RandomState(42)
                cluster_labels = rng.randint(0, actual_clusters, size=n_samples)

        # Intra-cluster dedup: precompute cluster similarity matrix, then greedy
        keep_set = set()
        duplicates = 0

        for cluster_id in range(actual_clusters):
            members = np.where(cluster_labels == cluster_id)[0]
            if len(members) == 0:
                continue

            cluster_mat = matrix_normed[members]
            # Precompute full intra-cluster cosine similarity matrix (dense)
            sim_matrix = (cluster_mat @ cluster_mat.T).toarray()

            kept_mask = np.zeros(len(members), dtype=bool)
            for local_j in range(len(members)):
                if kept_mask.any():
                    # Check max similarity against all previously kept members
                    max_sim = sim_matrix[local_j, kept_mask].max()
                    if max_sim >= cosine_threshold:
                        duplicates += 1
                        continue
                kept_mask[local_j] = True
                keep_set.add(valid_indices[members[local_j]])

            if cluster_id % 10 == 0:
                log.info("semdedup: cluster %d/%d done, kept so far: %d",
                         cluster_id + 1, actual_clusters, len(keep_set))

        # Build output: keep samples without features + kept samples with features
        valid_set = set(valid_indices)
        outputs = []
        for i, sample in enumerate(dataset):
            if i not in valid_set:
                outputs.append(sample)
            elif i in keep_set:
                outputs.append(sample)

        self._trace.notes["dedup_strategy"] = "semdedup_sae_clustering"
        self._trace.cost.extra["duplicates_removed"] = duplicates
        self._trace.cost.extra["cosine_threshold"] = cosine_threshold
        self._trace.cost.extra["num_clusters"] = actual_clusters
        self._trace.cost.extra["samples_with_features"] = n_samples
        self._trace.cost.extra["d_sae"] = d_sae
        self._trace.cost.extra["removal_rate"] = round(
            duplicates / max(1, n_samples), 3
        )
        return outputs