def _word_tokens(text):
    return re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", str(text))


def _is_word(token):
    return bool(re.fullmatch(r"\w+(?:[-']\w+)*", str(token)))


def _route_signature_for_target(text, target_disease, target_aspect):
    disease_profile = disease_routing_profile(text)
    aspect_profile = aspect_routing_profile(text)

    disease_selected = disease_profile["selected"]
    aspect_selected = target_aspect in aspect_profile["selected_aspects"]

    return {
        "disease_profile": disease_profile,
        "aspect_profile": aspect_profile,
        "disease_selected": disease_selected,
        "aspect_selected": aspect_selected,
        "same_route": (
            disease_selected.casefold() == target_disease.casefold()
            and aspect_selected
        ),
    }


def _aspect_support_profile_for_ctra(text, original_branch):
    """
    Preserve the branch that produced the accepted aspect decision.
    """
    if original_branch == "lexical":
        hits = lexical_aspect_hits(text)
        max_hit = max(hits.values()) if hits else 1

        scores = {
            aspect: float(hits.get(aspect, 0) / max_hit)
            for aspect in ASPECT_KEYWORDS
        }

        return {
            "branch": "lexical",
            "scores": scores,
        }

    semantic = semantic_aspect_profile(text)

    return {
        "branch": "semantic",
        "scores": semantic["scores"],
    }


def _intervention_components(
    full_score,
    removed_score,
    full_margin,
    removed_margin,
):
    positive = (
        0.5 * max(0.0, full_score - removed_score)
        + 0.5 * max(0.0, full_margin - removed_margin)
    )

    opposite = (
        0.5 * max(0.0, removed_score - full_score)
        + 0.5 * max(0.0, removed_margin - full_margin)
    )

    return float(positive), float(opposite)


def _selected_aspect_cue_tokens(text, aspect):
    cue_tokens = set()

    text_lower = str(text).casefold()

    for cue in ASPECT_KEYWORDS.get(aspect, []):
        cue_lower = cue.casefold()

        if cue_lower in text_lower:
            cue_tokens.update(
                normalise_phrase(cue_lower).split()
            )


        if " " not in cue_lower:
            cue_doc = nlp(cue_lower)
            if len(cue_doc):
                cue_tokens.add(
                    cue_doc[0].lemma_.casefold()
                )

    return cue_tokens


def _token_role(token, text, selected_disease, selected_aspect):
    token_n = normalise_phrase(token)

    disease_words = set(
        normalise_phrase(selected_disease).split()
    )

    aspect_cues = _selected_aspect_cue_tokens(
        text,
        selected_aspect,
    )

    token_lemma = ""
    doc = nlp(str(token))
    if len(doc):
        token_lemma = doc[0].lemma_.casefold()

    if token_n in disease_words:
        return "disease-condition evidence"

    if token_n in aspect_cues or token_lemma in aspect_cues:
        return "clinical-aspect evidence"


    competing_disease_words = set()
    for disease in exact_disease_matches(text):
        if disease.casefold() != selected_disease.casefold():
            competing_disease_words.update(
                normalise_phrase(disease).split()
            )

    if token_n in competing_disease_words:
        return "competing-route evidence"


    hits = lexical_aspect_hits(text)
    if any(
        aspect != selected_aspect
        and (
            token_n in _selected_aspect_cue_tokens(text, aspect)
            or token_lemma in _selected_aspect_cue_tokens(text, aspect)
        )
        for aspect in hits
    ):
        return "competing-route evidence"

    return "contextual information"


def _correction_lookup(edits):
    lookup = {}

    for edit in edits or []:
        before = str(edit.get("from", "")).strip()
        after = str(edit.get("to", "")).strip()

        if before and after:
            lookup[normalise_phrase(after)] = before

    return lookup


def ctra_route_explanation(route):
    """
    Explain one accepted disease--aspect route.
    """
    if not route.get("accepted"):
        raise ValueError("CTRA can only explain an accepted route.")

    text = str(route["accepted_text"])
    selected_disease = route["disease"]
    selected_aspect = route["aspect"]

    tokens = _word_tokens(text)

    baseline_disease_profile = disease_routing_profile(text)
    active_disease_sources = baseline_disease_profile["active_sources"]

    baseline_aspect_profile = route.get(
        "aspect_profile",
        aspect_routing_profile(text),
    )
    aspect_branch = baseline_aspect_profile["branch"]


    s_d_full, m_d_full = _selected_margin(
        baseline_disease_profile["scores"],
        selected_disease,
    )


    a_full_profile = _aspect_support_profile_for_ctra(
        text,
        aspect_branch,
    )
    s_a_full, m_a_full = _selected_margin(
        a_full_profile["scores"],
        selected_aspect,
    )

    rows = []

    for index, token in enumerate(tokens):
        if not _is_word(token):
            rows.append({
                "index": index,
                "token": token,
                "punctuation": True,
                "I": 0.0,
                "B": 0,
                "role": "punctuation",
            })
            continue

        reduced_tokens = (
            tokens[:index]
            + tokens[index + 1:]
        )

        reduced_text = " ".join(reduced_tokens)

        d_removed = disease_routing_profile(
            reduced_text,
            active_sources=active_disease_sources,
        )

        s_d_removed, m_d_removed = _selected_margin(
            d_removed["scores"],
            selected_disease,
        )

        p_d, n_d = _intervention_components(
            s_d_full,
            s_d_removed,
            m_d_full,
            m_d_removed,
        )

        a_removed = _aspect_support_profile_for_ctra(
            reduced_text,
            aspect_branch,
        )

        s_a_removed, m_a_removed = _selected_margin(
            a_removed["scores"],
            selected_aspect,
        )

        p_a, n_a = _intervention_components(
            s_a_full,
            s_a_removed,
            m_a_full,
            m_a_removed,
        )

        signed = (
            (p_d + p_a)
            -
            (n_d + n_a)
        )

        actual_removed_route = _route_signature_for_target(
            reduced_text,
            selected_disease,
            selected_aspect,
        )

        load_bearing = int(
            not actual_removed_route["same_route"]
        )

        rows.append({
            "index": index,
            "token": token,
            "punctuation": False,
            "P_D": p_d,
            "N_D": n_d,
            "P_A": p_a,
            "N_A": n_a,
            "I": float(signed),
            "B": load_bearing,
            "role": _token_role(
                token,
                text,
                selected_disease,
                selected_aspect,
            ),
        })


    structural_roles = {
        "disease-condition evidence",
        "clinical-aspect evidence",
    }

    groups = defaultdict(list)

    for row in rows:
        if (
            not row["punctuation"]
            and row["role"] in structural_roles
        ):
            groups[normalise_phrase(row["token"])].append(row)

    for _, group_rows in groups.items():
        if len(group_rows) <= 1:
            continue

        remove_indices = {
            row["index"]
            for row in group_rows
        }

        reduced_tokens = [
            token
            for idx, token in enumerate(tokens)
            if idx not in remove_indices
        ]

        reduced_text = " ".join(reduced_tokens)

        d_removed = disease_routing_profile(
            reduced_text,
            active_sources=active_disease_sources,
        )

        s_d_removed, m_d_removed = _selected_margin(
            d_removed["scores"],
            selected_disease,
        )

        p_d, n_d = _intervention_components(
            s_d_full,
            s_d_removed,
            m_d_full,
            m_d_removed,
        )

        a_removed = _aspect_support_profile_for_ctra(
            reduced_text,
            aspect_branch,
        )

        s_a_removed, m_a_removed = _selected_margin(
            a_removed["scores"],
            selected_aspect,
        )

        p_a, n_a = _intervention_components(
            s_a_full,
            s_a_removed,
            m_a_full,
            m_a_removed,
        )

        group_i = (
            (p_d + p_a)
            -
            (n_d + n_a)
        )

        sensitivities = np.asarray(
            [abs(row["I"]) for row in group_rows],
            dtype=float,
        )

        if sensitivities.sum() <= EPS:
            proportions = np.ones(
                len(group_rows),
                dtype=float,
            ) / len(group_rows)
        else:
            proportions = (
                sensitivities
                /
                sensitivities.sum()
            )

        actual_removed_route = _route_signature_for_target(
            reduced_text,
            selected_disease,
            selected_aspect,
        )

        group_b = int(
            not actual_removed_route["same_route"]
        )

        for row, proportion in zip(
            group_rows,
            proportions,
        ):
            row["I"] = float(
                group_i * proportion
            )
            row["B"] = max(
                row["B"],
                group_b,
            )
            row["grouped_repeated_evidence"] = True

    return {
        "disease": selected_disease,
        "aspect": selected_aspect,
        "text": text,
        "segment_index": route.get("segment_index"),
        "rows": rows,
    }


def ctra_explain_answer(answer_result):
    """
    Apply CTRA independently to every accepted route, then align token effects
    for presentation and use one common denominator across the complete query.
    """
    route_explanations = [
        ctra_route_explanation(route)
        for route in answer_result["accepted_routes"]
    ]

    aligned = {}

    for route_id, explanation in enumerate(route_explanations):
        for row in explanation["rows"]:
            if row["punctuation"]:
                continue

            key = (
                explanation.get("segment_index"),
                row["index"],
                normalise_phrase(row["token"]),
            )

            if key not in aligned:
                aligned[key] = {
                    "segment_index": explanation.get("segment_index"),
                    "index": row["index"],
                    "token": row["token"],
                    "I": 0.0,
                    "B": 0,
                    "roles": set(),
                    "routes": [],
                }

            aligned[key]["I"] += float(
                row.get("I", 0.0)
            )
            aligned[key]["B"] = max(
                aligned[key]["B"],
                int(row.get("B", 0)),
            )
            aligned[key]["roles"].add(
                row.get("role", "contextual information")
            )
            aligned[key]["routes"].append({
                "disease": explanation["disease"],
                "aspect": explanation["aspect"],
            })

    denominator = sum(
        abs(item["I"])
        for item in aligned.values()
    )

    correction_lookup = _correction_lookup(
        answer_result.get("grammar", {}).get("edits", [])
    )

    output_rows = []

    for item in sorted(
        aligned.values(),
        key=lambda row: (
            -1 if row["segment_index"] is None else row["segment_index"],
            row["index"],
        ),
    ):
        item = dict(item)

        item["W"] = (
            100.0 * abs(item["I"]) / denominator
            if denominator > EPS
            else 0.0
        )

        role_order = [
            "disease-condition evidence",
            "clinical-aspect evidence",
            "competing-route evidence",
            "contextual information",
        ]

        item["roles"] = [
            role
            for role in role_order
            if role in item["roles"]
        ]

        token_key = normalise_phrase(item["token"])
        item["was_corrected"] = token_key in correction_lookup
        item["original_form"] = correction_lookup.get(token_key)

        output_rows.append(item)

    return {
        "routes": route_explanations,
        "aligned_rows": output_rows,
        "weight_sum": sum(row["W"] for row in output_rows),
    }


def display_ctra_table(ctra_result):
    frame = pd.DataFrame([
        {
            "Token": row["token"],
            "Role": " + ".join(row["roles"]),
            "Signed effect I": row["I"],
            "Load-bearing B": row["B"],
            "Impact Weight (%)": row["W"],
            "Corrected": row["was_corrected"],
            "Original form": row["original_form"],
        }
        for row in ctra_result["aligned_rows"]
    ])

    return frame


print("CTRA implementation ready.")
