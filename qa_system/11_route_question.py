def route_segment_once(routing_text):
    disease_profile = disease_routing_profile(routing_text)
    aspect_profile = aspect_routing_profile(routing_text)

    selected_disease = disease_profile["selected"]
    c_d = disease_support(
        disease_profile,
        selected_disease,
    )

    routes = []

    for aspect in aspect_profile["selected_aspects"]:
        c_a = aspect_support(
            aspect_profile,
            aspect,
        )

        c_r = float((c_d + c_a) / 2.0)

        routes.append({
            "disease": selected_disease,
            "aspect": aspect,
            "C_D": c_d,
            "C_A": c_a,
            "C_R": c_r,
            "accepted": bool(c_r >= ROUTING_THRESHOLD),
            "disease_profile": disease_profile,
            "aspect_profile": aspect_profile,
            "accepted_text": str(routing_text),
            "recovered_by_t5": False,
        })

    return routes


def route_question_once(raw_query):
    grammar = grammar_correct(raw_query)
    corrected = grammar["corrected"]
    segments = controlled_query_segments(corrected)

    route_rows = []

    for segment_index, segment in enumerate(segments):
        segment_routes = route_segment_once(
            segment["routing_text"]
        )

        for route in segment_routes:
            route.update({
                "segment_index": segment_index,
                "segment_text": segment["text"],
                "segment_routing_text": segment["routing_text"],
                "carried_disease": segment["carried_disease"],
                "grammar_edits": grammar["edits"],
                "grammar_backend": grammar["backend"],
            })
            route_rows.append(route)


    merged = {}

    for route in route_rows:
        key = (
            route["disease"].casefold(),
            route["aspect"].casefold(),
        )

        if (
            key not in merged
            or route["C_R"] > merged[key]["C_R"]
        ):
            merged[key] = route

    return {
        "original_query": str(raw_query),
        "corrected_query": corrected,
        "grammar": grammar,
        "segments": segments,
        "routes": list(merged.values()),
    }
