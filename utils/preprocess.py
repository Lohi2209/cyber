import re


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
EMOTICON_PATTERN = re.compile(r"(?::|;|=)(?:-)?(?:\)|\(|d|p)+", re.IGNORECASE)
REPEATED_CHARACTER_PATTERN = re.compile(r"(.)\1{2,}")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")

EMOTICON_REPLACEMENTS = {
    ":)": " positive ",
    ":-)": " positive ",
    ":(": " negative ",
    ":-(": " negative ",
    ";)": " positive ",
    ";-)": " positive ",
    ":d": " positive ",
    ":-d": " positive ",
    ":p": " playful ",
    ":-p": " playful ",
}

SLANG_REPLACEMENTS = {
    "ur": "your",
    "u": "you",
    "r": "are",
    "y": "why",
    "stfu": "shut up",
    "wtf": "what the fuck",
}


def _replace_slang(text: str) -> str:
    words = text.split()
    return " ".join(SLANG_REPLACEMENTS.get(word, word) for word in words)


def clean_text(text: str) -> str:
    if text is None:
        return ""

    cleaned = str(text).lower()
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = MENTION_PATTERN.sub(" ", cleaned)
    cleaned = HASHTAG_PATTERN.sub(r"\1", cleaned)
    for emoticon, replacement in EMOTICON_REPLACEMENTS.items():
        cleaned = cleaned.replace(emoticon, replacement)
    cleaned = EMOTICON_PATTERN.sub(" ", cleaned)
    cleaned = _replace_slang(cleaned)
    cleaned = REPEATED_CHARACTER_PATTERN.sub(r"\1\1", cleaned)
    cleaned = NON_ALPHA_PATTERN.sub(" ", cleaned)
    cleaned = MULTISPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def add_basic_text_features(dataframe, column_name: str):
    features = dataframe.copy()
    text_series = features[column_name].fillna("").astype(str)
    features["message_length"] = text_series.str.len()
    features["word_count"] = text_series.str.split().str.len()
    features["capital_letters"] = text_series.apply(lambda value: sum(character.isupper() for character in value))
    features["punctuation_count"] = text_series.apply(lambda value: sum(character in "!?.,;:" for character in value))
    features["question_marks"] = text_series.str.count(r"\?")
    features["exclamation_marks"] = text_series.str.count(r"!")
    return features
