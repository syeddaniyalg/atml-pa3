import re

CONTRACTIONS = {
    "dont": "don't", "doesnt": "doesn't", "isnt": "isn't", "arent": "aren't", "wasnt": "wasn't",
    "werent": "weren't", "cant": "can't", "couldnt": "couldn't", "wouldnt": "wouldn't",
    "shouldnt": "shouldn't", "wont": "won't", "didnt": "didn't", "hasnt": "hasn't",
    "havent": "haven't", "hadnt": "hadn't", "im": "i'm", "ive": "i've", "youre": "you're",
    "thats": "that's", "whats": "what's", "theres": "there's", "heres": "here's",
    "hes": "he's", "shes": "she's", "its": "it's", "theyre": "they're", "lets": "let's",
}
NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
ARTICLES = {"a", "an", "the"}
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def _punctuation(text):
    out = text
    for p in PUNCT:
        if (p + " " in text or " " + p in text) or re.search(COMMA_STRIP, text):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return PERIOD_STRIP.sub("", out)


def _digits_articles(text):
    words = []
    for word in text.lower().split():
        word = NUMBER_MAP.get(word, word)
        if word not in ARTICLES:
            words.append(word)
    return " ".join(CONTRACTIONS.get(w, w) for w in words)


def normalize(answer):
    answer = answer.replace("\n", " ").replace("\t", " ").strip()
    return _digits_articles(_punctuation(answer))


def soft_accuracy(pred, refs):
    p = normalize(pred)
    ref = [normalize(r) for r in refs]
    scores = []
    for i in range(len(ref)):
        others = sum(1 for j, r in enumerate(ref) if j != i and r == p)
        scores.append(min(1.0, others / 3.0))
    return sum(scores) / len(scores)


def exact_match(pred, target):
    return float(normalize(pred) == normalize(target))
