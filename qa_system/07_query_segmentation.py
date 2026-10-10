_SEGMENT_SEPARATOR = re.compile(
    r"(?<=[.!?])\s+|,\s*|\s+\b(?:and|or)\b\s+",
    flags=re.IGNORECASE,
)


def _fragment_spans(text):
    text = str(text)
    fragments = []
    start = 0

    for match in _SEGMENT_SEPARATOR.finditer(text):
        fragment = text[start:match.start()].strip()
        if fragment:
            left = text.find(fragment, start, match.start() + 1)
            fragments.append((fragment, left, left + len(fragment)))
        start = match.end()

    tail = text[start:].strip()
    if tail:
        left = text.find(tail, start)
        fragments.append((tail, left, left + len(tail)))

    return fragments


def _disease_mentions_with_spans(text):
    mentions = []

    for disease in diseases:
        for match in re.finditer(
            re.escape(disease),
            str(text),
            flags=re.IGNORECASE,
        ):
            mentions.append({
                "disease": disease,
                "start": match.start(),
                "end": match.end(),
                "centre": (match.start() + match.end()) / 2.0,
            })

    return mentions


def controlled_query_segments(corrected_query):
    corrected_query = str(corrected_query).strip()

    fragments = _fragment_spans(corrected_query)


    if len(fragments) <= 1:
        return [{
            "text": corrected_query,
            "routing_text": corrected_query,
            "carried_disease": None,
        }]


    if not all(lexical_aspect_hits(fragment) for fragment, _, _ in fragments):
        return [{
            "text": corrected_query,
            "routing_text": corrected_query,
            "carried_disease": None,
        }]

    disease_mentions = _disease_mentions_with_spans(corrected_query)
    output = []

    for fragment, start, end in fragments:
        explicit = exact_disease_matches(fragment)

        if explicit:
            carried = None
            routing_text = fragment
        elif disease_mentions:
            centre = (start + end) / 2.0
            nearest = min(
                disease_mentions,
                key=lambda item: abs(item["centre"] - centre),
            )
            carried = nearest["disease"]
            routing_text = f"{carried} {fragment}".strip()
        else:
            carried = None
            routing_text = fragment

        output.append({
            "text": fragment,
            "routing_text": routing_text,
            "carried_disease": carried,
        })

    return output
