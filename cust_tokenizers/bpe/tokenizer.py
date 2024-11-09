from cust_tokenizers.shared_data.consts import *

from collections import defaultdict, Counter
from typing import List, Union
import numpy as np
import re
from tqdm import tqdm


class BPETokenizer:
    def __init__(self, vocab_size=10_000, lowercase=True, verbose=False):
        """
            Construnct BPE tokenizer

            Args:
                `vocab_size (int)`: size of vocabulary
                `lowercase (bool)`: if True, all text will be lowercased
                `verbose (bool)`: if True, log info will be shown
        """

        self._util_tokens = [UNK_TOKEN, PAD_TOKEN, SOW_TOKEN, EOW_TOKEN]

        # max size of vocabulary
        self.vocab_size = vocab_size + len(self._util_tokens)

        # maps token to index
        self._vocab = dict()

        for util_token in self._util_tokens:
            self._vocab[util_token] = len(self._vocab)

        # maps index to token
        self._reverse_vocab = dict()

        # regex to split text into tokens
        self._pattern = re.compile('\w+|[^\w\s]+')

        self._lowercase = lowercase
        self._verbose = verbose
        self._progress_bar = tqdm if verbose else iter

    def encode_2d(self, texts: Union[str, List[str]], word_size=None, text_size=None):
        if isinstance(texts, str):
            texts = [texts]

        if self._lowercase:
            texts = [text.lower() for text in texts]

        input_ids = []
        attention_mask = []

        # dunno how to name it properly
        # it is maximum of all max_sizes
        max_of_maxes = -(LARGE_INT - 1)

        for text in texts:
            # -1 to avoid overflow
            max_size = -(LARGE_INT - 1)

            text_ids = []
            text_mask = []
            tokens = self._pattern.findall(text)

            for token in tokens:
                split = [c for c in token]

                # use token pairs to efficiently encode token
                i = 0
                while i < len(split) - 1:
                    pair = split[i] + split[i + 1]
                    # pair = tuple([split[i], split[i + 1]])

                    # try to use token pair
                    if pair in self._vocab:
                        split[i: i + 2] = [pair]
                    else:
                        i += 1

                # make current token input_ids
                token_ids = [self._vocab[SOW_TOKEN]] + \
                    [self._vocab.get(c, self._vocab[UNK_TOKEN]) for c in split] + \
                    [self._vocab[EOW_TOKEN]]

                # add current token to attention mask
                text_mask.append([True] * len(token_ids))
                text_ids.append(token_ids)

                max_size = max(max_size, len(token_ids))

            input_ids.append([
                ids + [self._vocab.get(PAD_TOKEN, 1)] * (max_size - len(ids)) for ids in text_ids
            ])

            attention_mask.append([
                mask + [False] * (max_size - len(mask)) for mask in text_mask
            ])

            max_of_maxes = max(max_of_maxes, max_size)

        if word_size is not None:
            max_of_maxes = word_size

        # add padding to input_ids
        for inp_ids in input_ids:
            for ids in inp_ids:
                if len(ids) < max_of_maxes:
                    ids.extend([self._vocab.get(PAD_TOKEN, 1)]
                               * (max_of_maxes - len(ids)))
                else:
                    ids = ids[:max_of_maxes]

            if text_size is not None:
                if len(inp_ids) < text_size:
                    inp_ids.extend([
                        [self._vocab.get(PAD_TOKEN, 1)] * max_of_maxes,
                    ] * (text_size - len(inp_ids)))
                else:
                    inp_ids = inp_ids[:text_size]

        # add padding to attention mask
        for att_mask in attention_mask:
            for mask in att_mask:
                if len(mask) < max_of_maxes:
                    mask.extend([False] * (max_of_maxes - len(mask)))
                else:
                    mask = mask[:max_of_maxes]

            if text_size is not None:
                if len(att_mask) < text_size:
                    att_mask.extend([
                        [self._vocab.get(PAD_TOKEN, 1)] * max_of_maxes,
                    ] * (text_size - len(att_mask)))
                else:
                    att_mask = att_mask[:text_size]

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
        }

    def encode_plain(self, texts: Union[str, List[str]], text_size=None):
        if isinstance(texts, str):
            texts = [texts]

        if self._lowercase:
            texts = [text.lower() for text in texts]

        input_ids = []
        attention_mask = []

        max_size = 0

        for text in texts:
            text_ids = []
            text_mask = []
            tokens = self._pattern.findall(text)

            for token in tokens:
                split = [c for c in token]

                # use token pairs to efficiently encode token
                i = 0
                while i < len(split) - 1:
                    pair = split[i] + split[i + 1]

                    # try to use token pair
                    if pair in self._vocab:
                        split[i: i + 2] = [pair]
                    else:
                        i += 1

                # make current token input_ids
                token_ids = [self._vocab[SOW_TOKEN]] + \
                    [self._vocab.get(c, self._vocab[UNK_TOKEN]) for c in split] + \
                    [self._vocab[EOW_TOKEN]]

                # add current token to attention mask
                text_mask.extend([True] * len(token_ids))
                text_ids.extend(token_ids)

            max_size = max(max_size, len(text_ids))
            input_ids.append(text_ids)
            attention_mask.append(text_mask)

        if text_size is not None:
            max_size = text_size

        # add padding to input_ids or truncate
        for i in range(len(input_ids)):
            ids = input_ids[i]
            if len(ids) < max_size:
                input_ids[i].extend([self._vocab.get(PAD_TOKEN, 1)] * (max_size - len(ids)))
            else:
                input_ids[i] = ids[:max_size]

        # add padding to attention mask or truncate
        for mask in attention_mask:
            if len(mask) < max_size:
                mask.extend([False] * (max_size - len(mask)))
            else:
                mask = mask[:max_size]
        
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
        }

    def encode(self, texts: Union[str, List[str]], mode='plain', word_size=None, text_size=None):
        if mode == '2d':
            return self.encode_2d(texts, word_size, text_size)
        elif mode == 'plain':
            return self.encode_plain(texts, text_size)
        return None

    def decode(self, token_ids, attention_mask):
        """
            Decodes text from token ids and attention mask

            Args:
                `token_ids (Union[List[List[int]], List[List[List[int]]])`: array of token ids for each text
                `attention_mask (Union[List[List[bool]], List[List[List[bool]]])`: array of attention masks for each text

            Returns:
                `texts (List[str])`: list of texts
        """

        if isinstance(token_ids[0][0], int):
            token_ids = [token_ids]

        if isinstance(attention_mask[0][0], bool):
            attention_mask = [attention_mask]

        texts = []

        for text_ids, text_att in zip(token_ids, attention_mask):
            text = ""

            for word_ids, word_att in zip(text_ids, text_att):
                for id, att in zip(word_ids, word_att):
                    if not att:
                        break

                    if id == self._vocab[EOW_TOKEN]:
                        continue

                    if id == self._vocab[SOW_TOKEN]:
                        text += ' '
                        continue

                    text += self._reverse_vocab.get(id, UNK_TOKEN)

            texts.append(text)

        return texts

    def fit(self, texts: List[str]):
        if self._lowercase:
            texts = [text.lower() for text in texts]

        words = Counter()
        used_chars = set()

        if self._verbose:
            print("Splitting texts into words")

        # make map of [word, frequency in texts]
        for text in self._progress_bar(texts):
            tokens = self._pattern.findall(text)
            words.update(tokens)
            used_chars.update(*tokens)

        # save obtained chars to vocabulary
        for char in self._progress_bar(used_chars):
            self._vocab[char] = len(self._vocab)

        # maps word to its char-by-char split and frequency in the texts
        word_splits = {
            word: ([c for c in word], amt) for word, amt in self._progress_bar(words.items())
        }

        if self._verbose:
            pbar = tqdm(total=self.vocab_size - len(self._vocab),
                        desc="Making token pairs. Vocabulary size: ")

        # make token pairs
        while len(self._vocab) < self.vocab_size:
            pairs = defaultdict(int)

            if self._verbose:
                pbar.update(1)

            # count token pairs frequency
            for _, (split, amt) in word_splits.items():
                for i in range(len(split) - 1):
                    pairs[tuple(split[i: i + 2])] += amt

            if not pairs:
                break

            # get pair with max frequency
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            new_token = ''.join(best_pair)

            # save new token
            self._vocab[new_token] = len(self._vocab)

            # update word_splits
            for word in word_splits.keys():
                split, amt = word_splits[word]

                i = 0
                while i < len(split) - 1:
                    # try to update token pair
                    if tuple(split[i: i + 2]) == best_pair:
                        split[i: i + 2] = [new_token]
                    else:
                        i += 1

                word_splits[word] = (split, amt)

        if self._verbose:
            pbar.close()

        self._reverse_vocab = {v: k for k, v in self._vocab.items()}

    def saveVocab(self, file_path):
        with open(file_path, "w") as f:
            for token in self._vocab.keys():
                if token in self._util_tokens:
                    continue

                f.write(f"{token}\n")

    @staticmethod
    def fromVocab(file_path, vocab_size=10_000, lowercase=True, verbose=False):
        bpeTokenizer = BPETokenizer(vocab_size, lowercase, verbose)

        with open(file_path, "r") as f:
            for line in f:
                token = line.strip()
                cur_size = len(bpeTokenizer._vocab)
                bpeTokenizer._vocab[token] = cur_size
                bpeTokenizer._reverse_vocab[cur_size] = token

        return bpeTokenizer
