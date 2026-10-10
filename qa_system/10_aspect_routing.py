def semantic_aspect_profile(text):
    semantic = _semantic_candidate_profile(
        text,
        aspect_embeddings_cache,
    )

    ranked = sorted(
        semantic["combined"].items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "text": str(text),
        "selected": ranked[0][0],
        "ranked": ranked,
        "scores": semantic["combined"],
        "semantic": semantic,
    }


def aspect_routing_profile(text):
    hits = lexical_aspect_hits(text)

    if hits:
        ordered_hits = sorted(
            hits.items(),
            key=lambda item: (-item[1], item[0]),
        )

        selected_aspects = [
            aspect
            for aspect, _ in ordered_hits
        ]

        max_hit = max(hits.values())

        lexical_scores = {
            aspect: float(hits.get(aspect, 0) / max_hit)
            for aspect in ASPECT_KEYWORDS
        }


        semantic = semantic_aspect_profile(text)

        remaining = [
            item
            for item in semantic["ranked"]
            if item[0] not in selected_aspects
        ]

        ranked = (
            [(aspect, lexical_scores[aspect]) for aspect in selected_aspects]
            + remaining
        )

        return {
            "text": str(text),
            "branch": "lexical",
            "selected_aspects": selected_aspects,
            "hits": hits,
            "scores": lexical_scores,
            "ranked": ranked,
            "semantic_diagnostic": semantic,
        }

    semantic = semantic_aspect_profile(text)

    return {
        "text": str(text),
        "branch": "semantic",
        "selected_aspects": [semantic["selected"]],
        "hits": {},
        "scores": semantic["scores"],
        "ranked": semantic["ranked"],
        "semantic_diagnostic": semantic,
    }


def aspect_support(profile, selected_aspect):
    if profile["branch"] == "lexical":
        return 1.0

    _, margin = _selected_margin(
        profile["scores"],
        selected_aspect,
    )

    sigma = float(np.std(list(profile["scores"].values())))

    return float(
        math.tanh(
            max(0.0, margin) / (sigma + EPS)
        )
    )
