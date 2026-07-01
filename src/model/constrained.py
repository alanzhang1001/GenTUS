import torch
from transformers import LogitsProcessor


class TrieNode:
    def __init__(self):
        self.children: dict[int, "TrieNode"] = {}
        self.is_end: bool = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, seq: list[int]):
        node = self.root
        for tok in seq:
            if tok not in node.children:
                node.children[tok] = TrieNode()
            node = node.children[tok]
        node.is_end = True

    def next_tokens(self, prefix: list[int]) -> list[int]:
        """Return valid next tokens given a generated prefix (excluding special tokens)."""
        node = self.root
        for tok in prefix:
            if tok not in node.children:
                return []
            node = node.children[tok]
        return list(node.children.keys())

    def is_complete(self, prefix: list[int]) -> bool:
        """Return whether the prefix is itself a complete inserted sequence."""
        node = self.root
        for tok in prefix:
            if tok not in node.children:
                return False
            node = node.children[tok]
        return node.is_end


class SemanticIDConstrainedDecoder(LogitsProcessor):
    """Force the decoder to only generate tokens that form a valid Semantic ID prefix."""

    def __init__(self, trie: Trie, valid_token_ids: set[int], eos_token_id: int):
        self.trie = trie
        self.valid_token_ids = valid_token_ids
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for i, seq in enumerate(input_ids):
            # Extract only the semantic-id tokens generated so far (skip BOS/prompt tokens)
            generated = [t.item() for t in seq if t.item() in self.valid_token_ids]
            allowed = self.trie.next_tokens(generated)
            if self.trie.is_complete(generated):
                allowed = [*allowed, self.eos_token_id]

            mask = torch.full_like(scores[i], float("-inf"))
            if allowed:
                mask[allowed] = 0.0
            else:
                # Sequence complete — only allow EOS
                mask[self.eos_token_id] = 0.0
            scores[i] = scores[i] + mask
        return scores


def _codeword_to_str(codeword: int) -> str:
    # Negative values are collision suffixes encoded as -(suffix+1)
    if codeword < 0:
        return f"[SFX{-(codeword + 1)}]"
    return str(codeword)


def build_trie_from_ids(all_sids: list[tuple], tokenizer) -> tuple["Trie", set[int]]:
    """Convert Semantic IDs to token-id sequences and insert into a Trie."""
    trie = Trie()
    valid_token_ids = set()
    for sid in all_sids:
        token_seq = []
        for codeword in sid:
            tok_ids = tokenizer.encode(_codeword_to_str(codeword), add_special_tokens=False)
            token_seq.extend(tok_ids)
            valid_token_ids.update(tok_ids)
        trie.insert(token_seq)
    return trie, valid_token_ids
