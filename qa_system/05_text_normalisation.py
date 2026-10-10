def normalise_phrase(text):
    text = str(text).casefold()
    text = re.sub(r"[_/]+", " ", text)
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _contains_complete_phrase(text, phrase):
    text_n = normalise_phrase(text)
    phrase_n = normalise_phrase(phrase)
    if not phrase_n:
        return False
    return bool(
        re.search(
            rf"(?<!\w){re.escape(phrase_n)}(?!\w)",
            text_n,
        )
    )


def exact_disease_matches(text):
    matches = [
        disease
        for disease in diseases
        if _contains_complete_phrase(text, disease)
    ]

    return sorted(matches, key=lambda x: (-len(normalise_phrase(x)), x.casefold()))


def _lemma_tokens(text):
    return [
        token.lemma_.casefold()
        for token in nlp(str(text))
        if not token.is_space
    ]


def lexical_aspect_hits(text):
    text_lower = str(text).casefold()
    lemmas = _lemma_tokens(text)
    counts = {}

    for aspect, cues in ASPECT_KEYWORDS.items():
        score = 0

        for cue in cues:
            cue_lower = cue.casefold()

            if " " in cue_lower:
                score += text_lower.count(cue_lower)
            else:
                cue_doc = nlp(cue_lower)
                cue_lemma = (
                    cue_doc[0].lemma_.casefold()
                    if len(cue_doc)
                    else cue_lower
                )
                score += lemmas.count(cue_lemma)

        if score > 0:
            counts[aspect] = int(score)

    return counts
