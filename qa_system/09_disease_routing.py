def _source_discrimination(score_map):
    values = sorted(score_map.values(), reverse=True)
    top1 = values[0]
    top2 = values[1] if len(values) > 1 else top1
    gap = max(0.0, float(top1 - top2))
    sigma = float(np.std(list(score_map.values())))
    delta = math.tanh(gap / (sigma + EPS))
    return delta, gap, sigma


def disease_routing_profile(text, active_sources=None):
    text = str(text)
    text_n = normalise_phrase(text)

    semantic = _semantic_candidate_profile(
        text,
        disease_embeddings_cache,
    )

    exact_matches = exact_disease_matches(text)


    exact_anchor = exact_matches[0] if exact_matches else None

    exact_scores = {
        disease: float(disease == exact_anchor)
        for disease in diseases
    }

    lexical_scores = {
        disease: fuzz.token_set_ratio(
            text_n,
            normalise_phrase(disease),
        ) / 100.0
        for disease in diseases
    }

    semantic_scores = semantic["combined"]

    evidence = {
        "exact": exact_scores,
        "lexical": lexical_scores,
        "semantic": semantic_scores,
    }

    if active_sources is None:
        active_sources = ["lexical", "semantic"]
        if exact_anchor is not None:
            active_sources = ["exact"] + active_sources
    else:
        active_sources = list(active_sources)

    deltas = {}
    diagnostics = {}

    for source in active_sources:
        delta, gap, sigma = _source_discrimination(
            evidence[source]
        )
        deltas[source] = delta
        diagnostics[source] = {
            "gap": gap,
            "std": sigma,
            "delta": delta,
        }

    denominator = sum(deltas.values())

    if denominator > EPS:
        fusion_weights = {
            source: deltas[source] / denominator
            for source in active_sources
        }
    else:
        equal = 1.0 / len(active_sources)
        fusion_weights = {
            source: equal
            for source in active_sources
        }

    hybrid = {
        disease: float(
            sum(
                fusion_weights[source] * evidence[source][disease]
                for source in active_sources
            )
        )
        for disease in diseases
    }

    ranked = sorted(
        hybrid.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "text": text,
        "selected": ranked[0][0],
        "ranked": ranked,
        "scores": hybrid,
        "active_sources": active_sources,
        "fusion_weights": fusion_weights,
        "source_diagnostics": diagnostics,
        "exact_matches": exact_matches,
        "exact_anchor": exact_anchor,
        "lexical_scores": lexical_scores,
        "semantic": semantic,
    }


def _selected_margin(score_map, selected):
    selected_score = float(score_map.get(selected, 0.0))

    competitors = [
        float(score)
        for candidate, score in score_map.items()
        if candidate != selected
    ]

    runner = max(competitors) if competitors else 0.0
    margin = selected_score - runner

    return selected_score, float(margin)


def disease_support(profile, selected=None):
    selected = selected or profile["selected"]
    _, margin = _selected_margin(
        profile["scores"],
        selected,
    )

    sigma = float(np.std(list(profile["scores"].values())))

    support = math.tanh(
        max(0.0, margin) / (sigma + EPS)
    )

    return float(support)
