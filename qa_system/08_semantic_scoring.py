def _cosine(a, b):
    return float(
        cosine_similarity(
            [np.asarray(a)],
            [np.asarray(b)],
        )[0][0]
    )


def _minmax_score_map(score_map):
    values = np.asarray(list(score_map.values()), dtype=float)
    lo = float(values.min())
    hi = float(values.max())

    return {
        key: float((value - lo) / (hi - lo + EPS))
        for key, value in score_map.items()
    }


def _adaptive_encoder_weights(normalised_scores):
    deltas = {}
    raw_gaps = {}
    stds = {}

    for encoder, score_map in normalised_scores.items():
        values = sorted(score_map.values(), reverse=True)
        top1 = values[0]
        top2 = values[1] if len(values) > 1 else top1

        gap = max(0.0, float(top1 - top2))
        sigma = float(np.std(list(score_map.values())))

        z = gap / (sigma + EPS)
        delta = math.tanh(z)

        raw_gaps[encoder] = gap
        stds[encoder] = sigma
        deltas[encoder] = delta

    mean_delta = float(np.mean(list(deltas.values())))

    unnormalised = {
        encoder: (
            ENCODER_PRIORS[encoder]
            * math.exp(BETA * (deltas[encoder] - mean_delta))
        )
        for encoder in normalised_scores
    }

    total = sum(unnormalised.values()) + EPS

    t = {
        encoder: value / total
        for encoder, value in unnormalised.items()
    }

    weights = {
        encoder: (
            (1.0 - ALPHA) * ENCODER_PRIORS[encoder]
            + ALPHA * t[encoder]
        )
        for encoder in normalised_scores
    }


    wsum = sum(weights.values()) + EPS
    weights = {
        encoder: value / wsum
        for encoder, value in weights.items()
    }

    return weights, {
        "raw_gaps": raw_gaps,
        "stds": stds,
        "deltas": deltas,
        "mean_delta": mean_delta,
    }


def _semantic_candidate_profile(text, candidate_embeddings):
    query_embeddings = encode_all(text)

    raw = {
        encoder: {}
        for encoder in ENCODER_PRIORS
    }

    for candidate, embeddings in candidate_embeddings.items():
        for encoder in raw:
            raw[encoder][candidate] = _cosine(
                query_embeddings[encoder],
                embeddings[encoder],
            )

    normalised = {
        encoder: _minmax_score_map(score_map)
        for encoder, score_map in raw.items()
    }

    weights, gate = _adaptive_encoder_weights(normalised)

    combined = {
        candidate: float(
            sum(
                weights[encoder] * normalised[encoder][candidate]
                for encoder in weights
            )
        )
        for candidate in candidate_embeddings
    }

    return {
        "raw": raw,
        "normalised": normalised,
        "weights": weights,
        "gate": gate,
        "combined": combined,
    }
