import os
import re
from collections import defaultdict
from typing import BinaryIO
from multiprocessing import Pool


class BPETokenizer:
    # GPT-2 regex pre-tokenizer
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    TOKENIZER = re.compile(PAT, re.IGNORECASE)
    SPECIAL_TOKEN = "<|endofdoc|>"

    def __init__(self):
        self.vocab: dict[tuple[int, ...], int] = {}
        self.inv_vocab: dict[int, tuple[int, ...]] = {}
        self.merges: list[tuple] = []
        self.next_id: int = 0

    # ---------- Chunking for parallel pre-tokenization ----------
    def find_chunk_boundaries(
        self, file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
    ) -> list[int]:
        """Find safe chunk boundaries in a file for parallel pre-tokenization."""
        assert isinstance(split_special_token, bytes), "split_special_token must be bytes"

        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096

        for bi in range(1, len(chunk_boundaries) - 1):
            pos = chunk_boundaries[bi]
            file.seek(pos)
            while True:
                mini_chunk = file.read(mini_chunk_size)
                if not mini_chunk:  # EOF
                    chunk_boundaries[bi] = file_size
                    break
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = pos + found_at
                    break
                pos += mini_chunk_size

        return sorted(set(chunk_boundaries))

    def _process_chunk(self, args):
        """Worker: tokenize file chunk and count token frequencies."""
        filename, start, end = args
        counts = defaultdict(int)

        with open(filename, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

        docs = re.split(re.escape(self.SPECIAL_TOKEN), chunk)
        for doc in docs:
            for match in self.TOKENIZER.finditer(doc):
                token = match.group()
                counts[token] += 1
        return counts

    def _merge_counts(self, dicts):
        merged = defaultdict(int)
        for d in dicts:
            for k, v in d.items():
                merged[k] += v
        return merged

    def run_pre_tokenization_parallel(self, filename: str, num_processes: int = 4):
        """Parallel pre-tokenization across file chunks."""
        with open(filename, "rb") as f:
            boundaries = self.find_chunk_boundaries(
                f, num_processes, self.SPECIAL_TOKEN.encode("utf-8")
            )

        args = [(filename, s, e) for s, e in zip(boundaries[:-1], boundaries[1:])]

        with Pool(num_processes) as pool:
            partial_counts = pool.map(self._process_chunk, args)

        return self._merge_counts(partial_counts)

    # ---------- Training ----------
    def get_pair_counts(self, freqs: dict[tuple[tuple[int, ...]], int]) -> dict[tuple[int, int], int]:
        """Count frequency of all adjacent byte pairs."""
        pair_counts = defaultdict(int)
        for sequence, count in freqs.items():
            for i in range(len(sequence) - 1):
                pair_counts[(sequence[i], sequence[i + 1])] += count
        return pair_counts

    def train(self, filename: str, vocab_size: int, num_processes: int = 4):
        """Train BPE merges and vocab from corpus."""
        pre_token_counts = self.run_pre_tokenization_parallel(filename, num_processes)

        # Initialize vocab with byte sequences
        freqs: dict[tuple[tuple[int, ...]], int] = {}
        for token, count in pre_token_counts.items():
            token_bytes = token.encode("utf-8")
            token_seq = tuple((b,) for b in token_bytes)
            freqs[token_seq] = count
            for b in token_seq:
                if b not in self.vocab:
                    self.vocab[b] = self.next_id
                    self.next_id += 1

        # Iteratively learn merges
        while len(self.vocab) < vocab_size:
            pair_counts = self.get_pair_counts(freqs)
            if not pair_counts:
                break

            max_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
            merged_token = max_pair[0] + max_pair[1]

            if merged_token in self.vocab:
                continue

            self.vocab[merged_token] = self.next_id
            self.next_id += 1
            self.merges.append(max_pair)

            # Replace max pair in sequences
            new_freqs = {}
            for seq, freq in freqs.items():
                new_seq = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i + 1]) == max_pair:
                        new_seq.append(merged_token)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_freqs[tuple(new_seq)] = freq
            freqs = new_freqs

        # Build inverse vocab for decoding
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

    # ---------- Encoding / Decoding ----------
    def encode(self, text: str) -> list[int]:
        """Encode text into BPE token IDs."""
        pre_tokens = re.findall(self.PAT, text)
        token_ids = []
        merge_set = set(self.merges)

        for token in pre_tokens:
            byte_seq = [(b,) for b in token.encode("utf-8")]

            while True:
                merged = False
                i = 0
                new_seq = []
                while i < len(byte_seq):
                    if i < len(byte_seq) - 1 and (byte_seq[i], byte_seq[i+1]) in merge_set:
                        new_seq.append(byte_seq[i] + byte_seq[i+1])
                        i += 2
                        merged = True
                    else:
                        new_seq.append(byte_seq[i])
                        i += 1
                byte_seq = new_seq
                if not merged:
                    break

            token_ids.extend(self.vocab[tok] for tok in byte_seq)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back into string."""
        byte_seq = []
        for tid in token_ids:
            tok_bytes = self.inv_vocab[tid]
            byte_seq.extend(tok_bytes if isinstance(tok_bytes, tuple) else [tok_bytes])

        # Convert list of ints back into bytes → decode utf-8
        return bytes(byte_seq).decode("utf-8", errors="ignore")