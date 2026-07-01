import json
import numpy as np


class SemanticIDManager:
    def __init__(self):
        self.table2id: dict[str, tuple] = {}
        self.id2table: dict[tuple, str] = {}
        # set of base tuples that had at least one collision
        self.collision_bases: set[tuple] = set()

    # Sentinel value: suffix integers are stored as negative numbers -(suffix+1)
    # so they are distinguishable from codebook codewords (always >= 0).
    @staticmethod
    def _encode_suffix(suffix: int) -> int:
        return -(suffix + 1)

    def build(self, table_names: list, indices: np.ndarray, embeddings: np.ndarray | None = None):
        """Build bidirectional mapping, resolving collisions by appending a suffix.

        When embeddings are provided, tables sharing the same base ID are sorted
        by their L2 distance to the group centroid before suffix assignment, so
        suffix order reflects embedding proximity rather than filesystem order.
        """
        self.table2id = {}
        self.id2table = {}
        self.collision_bases = set()

        # Group tables by base tuple
        from collections import defaultdict
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, idx in enumerate(indices):
            groups[tuple(idx.tolist())].append(i)

        for base, member_indices in groups.items():
            if len(member_indices) > 1:
                self.collision_bases.add(base)
                # Sort by distance to group centroid when embeddings are available
                if embeddings is not None:
                    group_embs = embeddings[member_indices]
                    centroid = group_embs.mean(axis=0)
                    dists = np.linalg.norm(group_embs - centroid, axis=1)
                    order = np.argsort(dists)
                    member_indices = [member_indices[i] for i in order]
                for suffix, i in enumerate(member_indices):
                    sid = base + (self._encode_suffix(suffix),)
                    self.table2id[table_names[i]] = sid
                    self.id2table[sid] = table_names[i]
            else:
                i = member_indices[0]
                self.table2id[table_names[i]] = base
                self.id2table[base] = table_names[i]

    def extend(self, table_names: list, indices: np.ndarray, embeddings: np.ndarray | None = None):
        """Append new table -> Semantic-ID mappings without changing existing IDs.

        This is used by online/full-pool experiments: the generator may already
        have been trained on the current IDs, so existing assignments must stay
        stable. New tables that collide with an existing base ID receive a
        collision suffix; existing unsuffixed IDs are not rewritten.
        """
        if len(table_names) != len(indices):
            raise ValueError(f"table_names/indices length mismatch: {len(table_names)} vs {len(indices)}")

        if indices is None or len(indices) == 0:
            return

        base_len = int(indices.shape[1]) if hasattr(indices, "shape") and len(indices.shape) > 1 else None
        if base_len is None:
            base_len = min((len(sid) for sid in self.table2id.values()), default=0)

        used_by_base: dict[tuple, set[int | None]] = {}
        for sid in self.table2id.values():
            base = tuple(sid[:base_len])
            suffix = None
            if len(sid) > base_len and sid[base_len] < 0:
                suffix = -(sid[base_len] + 1)
            used_by_base.setdefault(base, set()).add(suffix)

        new_groups: dict[tuple, list[int]] = {}
        for i, idx in enumerate(indices):
            name = table_names[i]
            if name in self.table2id:
                continue
            base = tuple(idx.tolist() if hasattr(idx, "tolist") else idx)
            new_groups.setdefault(base, []).append(i)

        for base, member_indices in sorted(new_groups.items(), key=lambda item: item[0]):
            if embeddings is not None and len(member_indices) > 1:
                group_embs = embeddings[member_indices]
                centroid = group_embs.mean(axis=0)
                dists = np.linalg.norm(group_embs - centroid, axis=1)
                order = np.argsort(dists)
                member_indices = [member_indices[i] for i in order]

            used_suffixes = used_by_base.setdefault(base, set())
            for i in member_indices:
                name = table_names[i]
                if name in self.table2id:
                    continue
                if not used_suffixes:
                    sid = base
                    used_suffixes.add(None)
                else:
                    numeric_suffixes = [s for s in used_suffixes if s is not None]
                    suffix = (max(numeric_suffixes) + 1) if numeric_suffixes else 0
                    while suffix in used_suffixes:
                        suffix += 1
                    sid = base + (self._encode_suffix(suffix),)
                    used_suffixes.add(suffix)
                    self.collision_bases.add(base)
                self.table2id[name] = sid
                self.id2table[sid] = name

        self.collision_bases.update(
            base for base, suffixes in used_by_base.items()
            if len(suffixes) > 1 or any(s is not None for s in suffixes)
        )

    def _base_sid(self, sid: tuple) -> tuple:
        if not self.table2id:
            return sid
        base_len = min(len(cur_sid) for cur_sid in self.table2id.values())
        return sid[:base_len]

    def table_to_id(self, name: str) -> tuple | None:
        return self.table2id.get(name)

    def id_to_table(self, sid: tuple) -> str | None:
        return self.id2table.get(sid)

    def all_ids(self) -> list:
        return list(self.id2table.keys())

    def save(self, path: str):
        data = {
            "table2id": {k: list(v) for k, v in self.table2id.items()},
            "id2table": {json.dumps(list(k)): v for k, v in self.id2table.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "SemanticIDManager":
        with open(path) as f:
            data = json.load(f)
        m = cls()
        m.table2id = {k: tuple(v) for k, v in data["table2id"].items()}
        m.id2table = {tuple(json.loads(k)): v for k, v in data["id2table"].items()}
        base_len = min(len(s) for s in m.table2id.values()) if m.table2id else 0
        m.collision_bases = {s[:base_len] for s in m.table2id.values() if len(s) > base_len}
        return m

    def collision_rate(self) -> float:
        n_tables = len(self.table2id)
        n_unique_base = len({self._base_sid(sid) for sid in self.table2id.values()})
        return 1.0 - n_unique_base / n_tables if n_tables else 0.0
