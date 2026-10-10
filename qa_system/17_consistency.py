def _accepted_route_set_once(question):
    """
    Route one formulation with GECToR + segmentation + disease/aspect routing,
    but without T5 recovery. Only routes satisfying 0.925 are returned.
    """
    result = route_question_once(question)

    accepted = [
        route
        for route in result["routes"]
        if route["accepted"]
    ]

    signature = tuple(sorted(
        (
            route["disease"].casefold(),
            route["aspect"].casefold(),
        )
        for route in accepted
    ))

    return result, accepted, signature


def _reference_result_for_consistency(question):
    """
    The reference route is the route actually accepted by the deployed system:
    original if accepted, otherwise the selected T5 recovery route(s).
    """
    answer = answer_question(
        question,
        retrieve=False,
        show=False,
    )

    signature = tuple(sorted(
        (
            route["disease"].casefold(),
            route["aspect"].casefold(),
        )
        for route in answer["accepted_routes"]
    ))

    return answer, signature


def route_preserving_alternatives(question):
    reference_answer, reference_signature = (
        _reference_result_for_consistency(question)
    )

    generated = generate_t5_reformulations(question)

    retained = []
    audit = []
    seen = set()

    for alternative in generated:
        key = re.sub(
            r"\s+",
            " ",
            alternative.strip().casefold(),
        )

        if key in seen:
            audit.append({
                "alternative": alternative,
                "status": "duplicate",
            })
            continue

        seen.add(key)

        result, accepted, signature = _accepted_route_set_once(
            alternative
        )

        all_candidate_routes_accepted = bool(
            result["routes"]
            and len(accepted) == len(result["routes"])
        )

        preserved = (
            bool(reference_signature)
            and all_candidate_routes_accepted
            and signature == reference_signature
        )

        if preserved:
            retained.append({
                "text": alternative,
                "result": result,
                "accepted_routes": accepted,
            })
            status = "route_preserving"
        elif not all_candidate_routes_accepted:
            status = "below_threshold_or_incomplete"
        else:
            status = "route_changed"

        audit.append({
            "alternative": alternative,
            "status": status,
            "signature": signature,
        })

    return {
        "reference_answer": reference_answer,
        "reference_signature": reference_signature,
        "generated": generated,
        "retained": retained,
        "audit": audit,
        "eligible": len(retained) >= 2,
    }


def explanation_consistency_for_question(question, method="CTRA"):
    probe = route_preserving_alternatives(question)

    if not probe["eligible"]:
        return {
            **probe,
            "consistency": np.nan,
            "pair_scores": [],
        }

    explanation_sets = []


    for item in probe["retained"]:


        words = []

        for route in item["accepted_routes"]:
            if method.upper() == "CTRA":
                words.extend(
                    ctra_selected_words_for_evaluation(route)
                )
            elif method.upper() == "LIME":
                words.extend(
                    lime_selected_words(
                        route["accepted_text"],
                        route["disease"],
                        route["aspect"],
                    )
                )
            else:
                raise ValueError("method must be 'CTRA' or 'LIME'")

        explanation_sets.append(words)

    pair_scores = [
        jaccard_explanation_consistency(
            explanation_sets[i],
            explanation_sets[j],
        )
        for i, j in combinations(
            range(len(explanation_sets)),
            2,
        )
    ]

    return {
        **probe,
        "consistency": (
            float(np.mean(pair_scores))
            if pair_scores
            else np.nan
        ),
        "pair_scores": pair_scores,
    }
