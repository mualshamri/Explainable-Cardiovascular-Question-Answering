try:
    from lime.lime_text import LimeTextExplainer
except Exception as exc:
    raise ImportError(
        "The 'lime' package is required for the CTRA-vs-LIME experiment."
    ) from exc


def _same_target_route(text, disease, aspect):
    routed = route_segment_once(text)

    return any(
        route["disease"].casefold() == disease.casefold()
        and route["aspect"].casefold() == aspect.casefold()
        for route in routed
    )


def lime_selected_words(text, disease, aspect):
    base_route = (disease, aspect)

    def predictor(texts):
        outputs = []

        for candidate in texts:
            preserved = _same_target_route(
                candidate,
                base_route[0],
                base_route[1],
            )

            p_preserved = 0.99 if preserved else 0.01

            outputs.append([
                1.0 - p_preserved,
                p_preserved,
            ])

        return np.asarray(outputs, dtype=float)

    explainer = LimeTextExplainer(
        class_names=[
            "route changed",
            "route preserved",
        ],
        random_state=LIME_RANDOM_STATE,
    )

    content_words = [
        token
        for token in _word_tokens(text)
        if _is_word(token)
    ]

    explanation = explainer.explain_instance(
        str(text),
        predictor,
        labels=[1],
        num_features=max(
            1,
            min(
                len(content_words),
                LIME_MAX_FEATURES,
            ),
        ),
        num_samples=LIME_NUM_SAMPLES,
    )

    weighted = explanation.as_list(label=1)

    positive = sorted(
        [
            (word, float(weight))
            for word, weight in weighted
            if float(weight) > 0.0
        ],
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    return [
        word
        for word, _ in positive[:LIME_TOP_POSITIVE]
    ]


def ctra_selected_words_for_evaluation(route):
    explanation = ctra_route_explanation(route)

    return [
        row["token"]
        for row in explanation["rows"]
        if (
            not row["punctuation"]
            and row["B"] == 1
        )
    ]


def _reduced_query_with_selected_words(text, selected_words):
    selected_counter = Counter(
        normalise_phrase(word)
        for word in selected_words
        if normalise_phrase(word)
    )

    kept = []

    for token in _word_tokens(text):
        key = normalise_phrase(token)

        if (
            _is_word(token)
            and selected_counter.get(key, 0) > 0
        ):
            kept.append(token)
            selected_counter[key] -= 1

    return " ".join(kept)


def explanation_sufficiency(text, disease, aspect, selected_words):
    reduced = _reduced_query_with_selected_words(
        text,
        selected_words,
    )

    if not reduced.strip():
        return 0

    return int(
        _same_target_route(
            reduced,
            disease,
            aspect,
        )
    )


def _canonical_concept_set(words):
    concepts = set()


    aspect_lemma_map = {}

    for aspect, cues in ASPECT_KEYWORDS.items():
        for cue in cues:
            for token in nlp(cue):
                if token.is_alpha:
                    aspect_lemma_map[
                        token.lemma_.casefold()
                    ] = aspect

    for word in words:
        doc = nlp(str(word))

        for token in doc:
            if not token.is_alpha:
                continue

            lemma = token.lemma_.casefold()

            concepts.add(
                aspect_lemma_map.get(
                    lemma,
                    lemma,
                )
            )

    return concepts


def jaccard_explanation_consistency(words_a, words_b):
    a = _canonical_concept_set(words_a)
    b = _canonical_concept_set(words_b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)
