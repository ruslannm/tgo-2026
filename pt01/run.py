"""Работа 1 заочного отделения. Задание: part-time/assignments/01/README.md в репозитории курса.

Запуск: python pt01/run.py, для CI python pt01/run.py --fast.
Артефакты bigram.json и bpe.json пишутся в pt01/artifacts/.
"""
import argparse
from pathlib import Path
import os
import math
import random
import re
import json
from collections import Counter
from urllib.request import urlretrieve
import numpy as np

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CORPUS = HERE / "corpus.txt"

SEED = 42
CORPUS_URL = "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"
WORD_RE = re.compile(r"[а-я]+(?:-[а-я]+)*")
MIN_WORD_FREQ = 2   # порог из задания не задан числом; 2 = отсечь гапаксы

def download_corpus():
    if not os.path.exists(CORPUS):
        print(f"Downloading corpus to {CORPUS} ...")
        urlretrieve(CORPUS_URL, CORPUS)
    with open(CORPUS, "r", encoding="utf-8") as f:
        return f.read()


def get_all_words(text):
    text = text.casefold().replace("ё", "е")
    return WORD_RE.findall(text)


class BigramLM:
    def __init__(self, alpha=0.01):
        self.context_length = 1
        self.alpha = alpha
        self.word_counts = Counter()
        self.bigram_counts = Counter()
        self.vocab = []
        self.vocab_set = set()
        self.vocab_size = 0
        self.total_words = 0

    def fit(self, words):
        self.word_counts = Counter(words)
        self.bigram_counts = Counter(zip(words, words[1:]))
        self.vocab = list(self.word_counts.keys())
        self.vocab_set = set(self.vocab)
        self.vocab_size = len(self.vocab)
        self.total_words = len(words)

    def prob(self, v, w):
        c_v = self.word_counts.get(v, 0)
        c_vw = self.bigram_counts.get((v, w), 0)
        return (c_vw + self.alpha) / (c_v + self.alpha * self.vocab_size)

    def logprob(self, words):
        if len(words) < 2:
            return 0.0
        total = 0.0
        for v, w in zip(words, words[1:]):
            total += math.log(self.prob(v, w))
        return total

    def generate(self, start, length=30, temperature=1.0, seed=SEED):
        if seed is not None:
            np.random.seed(seed)
        if start not in self.vocab_set:
           raise ValueError("Слова нет в словаре")
        result = [start]
        v = start
        for _ in range(length - 1):
            probs = np.array([self.prob(v, w) for w in self.vocab], dtype=np.float64)
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
            total = probs.sum()
            if total <= 0 or not np.isfinite(total):
                raise ValueError("Сумма вероятностей должна быть положительным числом")
            probs = probs / total
            v = self.vocab[np.random.choice(len(self.vocab), p=probs)]
            result.append(v)
        return result


class TrigramLM:
    def __init__(self, alpha=0.01, alpha_bigram=0.01):
        self.context_length = 2
        self.alpha = alpha
        self.bigram = BigramLM(alpha=alpha_bigram)   # для отката
        self.triple_counts = Counter()
        self.context_pair_counts = Counter()

    def fit(self, words):
        self.bigram.fit(words)
        self.triple_counts = Counter(zip(words, words[1:], words[2:]))
        self.context_pair_counts = Counter(zip(words, words[1:]))
        self.vocab = self.bigram.vocab
        self.vocab_set = self.bigram.vocab_set
        self.vocab_size = self.bigram.vocab_size

    def prob(self, u, v, w):
        if (u, v, w) not in self.triple_counts:
            return self.bigram.prob(v, w)      # откат
        c_uv = self.context_pair_counts.get((u, v), 0)
        c_uvw = self.triple_counts.get((u, v, w), 0)
        return (c_uvw + self.alpha) / (c_uv + self.alpha * self.vocab_size)

    def generate(self, start, length=30, temperature=1.0, seed=SEED):
        if seed is not None:
            np.random.seed(seed)
        if start not in self.vocab_set:
           raise ValueError("Слова нет в словаре")
        # Генерируем второе слово биграмной моделью
        result = self.bigram.generate(start, 2, temperature=temperature, seed=seed)
        u, v = result
        for _ in range(length - 2):
            probs = np.array([self.prob(u, v, w) for w in self.vocab], dtype=np.float64)
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
            total = probs.sum()
            if total <= 0 or not np.isfinite(total):
                raise ValueError("Сумма вероятностей должна быть положительным числом")
            probs = probs / total
            w = self.vocab[np.random.choice(len(self.vocab), p=probs)]
            result.append(w)
            u, v = v, w                    # сдвиг контекста
        return result


def get_generations(lm):
    generations = []
    for start in ["князь", "она"]:
        for temperature in [0.5, 1, 2]:
            words = lm.generate(start, length=30, temperature=temperature, seed=SEED)
            generations.append({
                "temperature": temperature,
                "start": start,
                "text": " ".join(words),
            })
    return generations


def get_likelihood(lm, words, sentence_len=12, alpha=0.01):
    # Likelihood comparison
    sentence = words[:sentence_len]
    shuffled = sentence[:]
    rng = random.Random(SEED)
    rng.shuffle(shuffled)
    # ensure shuffled differs from sentence
    if shuffled == sentence and len(sentence) > 1:
        shuffled = sentence[1:] + sentence[:1]    
    saved_alpha = lm.alpha
    lm.alpha = alpha
    logprob_sentence = lm.logprob(sentence)
    logprob_shuffled = lm.logprob(shuffled)
    unseen_pairs_in_sentence = sum(1 for v, w in zip(sentence, sentence[1:]) if (v, w) not in lm.bigram_counts)
    likelihood =  {
                "sentence": " ".join(sentence),
                "shuffled": " ".join(shuffled),
                "alpha": lm.alpha,
                "logprob_sentence": logprob_sentence,
                "logprob_shuffled": logprob_shuffled,
                "unseen_pairs_in_sentence": unseen_pairs_in_sentence,
            }
    lm.alpha = saved_alpha    
    return likelihood

def get_perplexity(lm, words, alpha_values):
    # Perplexity on validation for several alphas
    saved_alpha = lm.alpha
    perplexities = {}
    for alpha in alpha_values:
        lm.alpha = alpha
        nll = 0.0
        if lm.context_length == 1:
            for v, w in zip(words, words[1:]):
                nll -= math.log(lm.prob(v, w))
            count = len(words[1:])
        elif lm.context_length == 2:
            for u, v, w in zip(words, words[1:], words[2:]):
                nll -= math.log(lm.prob(u, v, w))
            count = len(words[2:])
        else:
            raise ValueError("Не определена длина контекста")
        perplexities[str(alpha)] = math.exp(nll / count) if count else float("inf")
    lm.alpha = saved_alpha 
    return perplexities

def compute_bigram_artifacts(all_words):
    n = len(all_words)
    split = int(n * 0.9)
    train_words = all_words[:split]
    val_words = all_words[split:]

    lm = BigramLM(alpha=0.01)
    lm.fit(train_words)

    top20 = [[w, c] for w, c in lm.word_counts.most_common(20)]

    generations = get_generations(lm)
    likelihood = get_likelihood(lm, val_words, sentence_len=12, alpha=0.01)
    alpha_values = [1, 0.1, 0.01]
    perplexity = get_perplexity(lm, val_words, alpha_values)
    bigram = {
            "corpus": CORPUS_URL,
            "words": n,
            "train_words": len(train_words),
            "val_words": len(val_words),
            "vocab_size": lm.vocab_size,
            "top20": top20,
            "perplexity": perplexity,
            "likelihood": likelihood,
            "generation": generations,
        }
    return bigram

def word_to_chars(word):
    return list(word) + ["</w>"]


def get_best_pair(word_counts, vocab):
    pair_counts = Counter()
    for w, symbols in vocab.items():
        f = word_counts[w]
        for a, b in zip(symbols, symbols[1:]):
            pair_counts[(a, b)] += f
    if not pair_counts:
        return None
    best_pair, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
    if best_count < 2:
        return None
    return best_pair


def learn_bpe(word_counts: dict, num_merges):
    # word_freqs: dict word -> freq (word is raw string)
    # Represent each word as list of symbols
    vocab = {w: word_to_chars(w) for w in word_counts}
    merges = []

    for _ in range(num_merges):
        best_pair = get_best_pair(word_counts, vocab)
        if best_pair is None:
            break
        a, b = best_pair
        new_symbol = a + b
        # Apply merge to all words
        for w, symbols in vocab.items():
            if a not in symbols:
                continue
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(new_symbol)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            vocab[w] = new_symbols
        merges.append([a, b])
    return merges


def apply_bpe(word, merge_ranks):
    symbols = word_to_chars(word)
    changed = True
    while changed and len(symbols) > 1:
        changed = False
        best_rank = None
        best_idx = -1
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            r = merge_ranks.get(pair)
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best_idx = i
        if best_rank is not None:
            a, b = symbols[best_idx], symbols[best_idx + 1]
            symbols[best_idx:best_idx + 2] = [a + b]
            changed = True
    return symbols


def compute_bpe_artifacts(all_words, num_merges):
    word_counts = Counter(all_words)
    filtered = {w: c for w, c in word_counts.items() if c >= MIN_WORD_FREQ}
    merges = learn_bpe(filtered, num_merges)
    merge_ranks = {tuple(m): i for i, m in enumerate(merges)}

    # Stats on all words (use raw word list, including low freq)
    chars = 0
    tokens_bpe = 0
    words = 0
    for w, c in word_counts.items():
        chars += len(w) * c
        words += c
        toks = apply_bpe(w, merge_ranks)
        tokens_bpe += len(toks) * c

    tokens_per_word = tokens_bpe / words if words else 0
    compression = chars / tokens_bpe if tokens_bpe else 0

    # Examples
    example_words = ["наполеон", "сказала", "неудовольствие"]
    examples = {}
    for w in example_words:
        examples[w] = apply_bpe(w, merge_ranks)

    # vocab size: number of distinct symbols (initial alphabet + merged symbols)
    alphabet = set()
    for w in word_counts:
        for ch in word_to_chars(w):
            alphabet.add(ch)
    vocab_size = len(alphabet) + len(merges)

    bpe = {
        "num_merges": len(merges),
        "vocab_size": vocab_size,
        "merges": merges,
        "stats": {
            "chars": chars,
            "words": words,
            "tokens_bpe": tokens_bpe,
            "tokens_per_word": tokens_per_word,
            "compression_vs_chars": compression,
        },
        "examples": examples,
    }
    return bpe

def backoff_rate_on_trajectory(lm, words):
    """Доля шагов траектории, где (u,v,w) отсутствует в triple_counts."""
    triples = list(zip(words, words[1:], words[2:]))
    if not triples:
        return 0.0, 0, 0
    unseen = sum(1 for (u, v, w) in triples if (u, v, w) not in lm.triple_counts)
    return unseen / len(triples), unseen, len(triples)


def get_lm_comparison(all_words):
    n = len(all_words)
    split = int(n * 0.9)
    train_words = all_words[:split]
    val_words = all_words[split:]

    alpha_values = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]
    
    bigram_lm = BigramLM(alpha=0.01)
    bigram_lm.fit(train_words)
    bigram_generations = get_generations(bigram_lm)
    bigram_perplexity = get_perplexity(bigram_lm, val_words, alpha_values)

    trigram_lm = TrigramLM(alpha=0.01)
    trigram_lm.fit(train_words)
    trigram_generations = get_generations(trigram_lm)
    trigram_perplexity = get_perplexity(trigram_lm, val_words, alpha_values)
    
    # Вычисляем долю откатов на биграмную модель
    for item in trigram_generations:
        rate, unseen, total = backoff_rate_on_trajectory(trigram_lm, item["text"].split())
        item["rate"] = rate
        item["unseen"] = unseen
        item["total"] = total

    result = {
                "perplexity": {"bigram": bigram_perplexity,
                               "trigram": trigram_perplexity,
                },
                "generation": { "bigram": bigram_generations,
                                "trigram": trigram_generations,
                }
            }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="быстрый режим для CI")
    ap.add_argument("--num_merges", help="Количество мержей", type=int)
    ap.add_argument("--bonus_1", action="store_true", help="Сравнение биграмную и триграмную модели: перплексия, генерация")
    args = ap.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)

    text = download_corpus()
    all_words = get_all_words(text)

    bigram = compute_bigram_artifacts(all_words)
    with open(ARTIFACTS / "bigram.json", "w", encoding="utf-8") as f:
        json.dump(bigram, f, ensure_ascii=False, indent=2)
    
    num_merges = 200 if args.fast else 1500
    if args.num_merges:
        num_merges = args.num_merges
    bpe = compute_bpe_artifacts(all_words, num_merges)
    with open(ARTIFACTS / "bpe.json", "w", encoding="utf-8") as f:
        json.dump(bpe, f, ensure_ascii=False, indent=2)

    # Бонус 1 Сравнение биграмную и триграмную модели: перплексия, генерация
    if args.bonus_1: 
        lm_comparison = get_lm_comparison(all_words)
        with open(ARTIFACTS / "lm_comparison.json", "w", encoding="utf-8") as f:
            json.dump(lm_comparison, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
