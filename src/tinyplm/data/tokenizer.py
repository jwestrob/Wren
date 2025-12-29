"""Amino acid tokenizer for protein sequences."""

from typing import Optional


class ProteinTokenizer:
    """Tokenizer for protein sequences.

    Vocabulary:
        - Special tokens: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
        - Standard amino acids: A=5, R=6, N=7, D=8, C=9, E=10, Q=11, G=12,
                                H=13, I=14, L=15, K=16, M=17, F=18, P=19,
                                S=20, T=21, W=22, Y=23, V=24
    """

    # Special tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    MASK_TOKEN = "[MASK]"

    PAD_ID = 0
    UNK_ID = 1
    CLS_ID = 2
    SEP_ID = 3
    MASK_ID = 4

    # Standard 20 amino acids (single letter codes)
    AMINO_ACIDS = "ARNDCEQGHILKMFPSTWYV"

    # Non-standard amino acids and their mappings
    # B = D or N (Asx), Z = E or Q (Glx), X = any, J = I or L
    # O = Pyrrolysine (rare), U = Selenocysteine (rare)
    NONSTANDARD_MAP = {
        "B": "D",  # Asx -> Asp
        "Z": "E",  # Glx -> Glu
        "J": "L",  # Xle -> Leu
        "O": "K",  # Pyrrolysine -> Lys (closest)
        "U": "C",  # Selenocysteine -> Cys (closest)
        "X": None,  # Unknown -> UNK token
    }

    def __init__(self, add_special_tokens: bool = True):
        """Initialize tokenizer.

        Args:
            add_special_tokens: Whether to add [CLS] and [SEP] during encoding.
        """
        self.add_special_tokens = add_special_tokens

        # Build vocabulary
        self._special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.CLS_TOKEN,
            self.SEP_TOKEN,
            self.MASK_TOKEN,
        ]
        self._vocab = self._special_tokens + list(self.AMINO_ACIDS)

        # Token to ID mapping
        self._token_to_id = {token: idx for idx, token in enumerate(self._vocab)}
        self._id_to_token = {idx: token for idx, token in enumerate(self._vocab)}

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self._vocab)

    @property
    def special_token_ids(self) -> list[int]:
        """Return list of special token IDs."""
        return [self.PAD_ID, self.UNK_ID, self.CLS_ID, self.SEP_ID, self.MASK_ID]

    def _normalize_aa(self, aa: str) -> str:
        """Normalize amino acid, handling non-standard codes."""
        aa = aa.upper()
        if aa in self.AMINO_ACIDS:
            return aa
        if aa in self.NONSTANDARD_MAP:
            mapped = self.NONSTANDARD_MAP[aa]
            return mapped if mapped else self.UNK_TOKEN
        return self.UNK_TOKEN

    def encode(
        self,
        sequence: str,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True,
    ) -> list[int]:
        """Encode a protein sequence to token IDs.

        Args:
            sequence: Amino acid sequence (single letter codes).
            max_length: Maximum length (including special tokens).
            padding: Whether to pad to max_length.
            truncation: Whether to truncate to max_length.

        Returns:
            List of token IDs.
        """
        # Normalize and convert to token IDs
        tokens = []
        for aa in sequence:
            normalized = self._normalize_aa(aa)
            token_id = self._token_to_id.get(normalized, self.UNK_ID)
            tokens.append(token_id)

        # Add special tokens
        if self.add_special_tokens:
            tokens = [self.CLS_ID] + tokens + [self.SEP_ID]

        # Truncation
        if truncation and max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
            # Ensure SEP is at the end if we truncated
            if self.add_special_tokens:
                tokens[-1] = self.SEP_ID

        # Padding
        if padding and max_length is not None:
            pad_length = max_length - len(tokens)
            if pad_length > 0:
                tokens = tokens + [self.PAD_ID] * pad_length

        return tokens

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to amino acid sequence.

        Args:
            token_ids: List of token IDs.
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            Amino acid sequence string.
        """
        tokens = []
        for token_id in token_ids:
            token = self._id_to_token.get(token_id, self.UNK_TOKEN)
            if skip_special_tokens and token in self._special_tokens:
                continue
            tokens.append(token)
        return "".join(tokens)

    def batch_encode(
        self,
        sequences: list[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> dict[str, list[list[int]]]:
        """Encode a batch of sequences.

        Args:
            sequences: List of amino acid sequences.
            max_length: Maximum length. If None, uses longest sequence.
            padding: Whether to pad to max_length.
            truncation: Whether to truncate to max_length.

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'.
        """
        # First pass: encode without padding to find max length
        encoded = [
            self.encode(seq, max_length=max_length, padding=False, truncation=truncation)
            for seq in sequences
        ]

        if max_length is None:
            max_length = max(len(ids) for ids in encoded)

        # Second pass: pad to uniform length
        input_ids = []
        attention_mask = []
        for ids in encoded:
            pad_length = max_length - len(ids)
            input_ids.append(ids + [self.PAD_ID] * pad_length)
            attention_mask.append([1] * len(ids) + [0] * pad_length)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def get_vocab(self) -> dict[str, int]:
        """Return the vocabulary as a token->id mapping."""
        return self._token_to_id.copy()
