def answer_question(raw_query, retrieve=True, show=True):
    direct = route_question_once(raw_query)

    final_routes = []

    for route in direct["routes"]:
        if route["accepted"]:
            final_routes.append(route)
        else:
            final_routes.append(
                recover_low_support_route(route)
            )


    merged = {}

    for route in final_routes:
        key = (
            route["disease"].casefold(),
            route["aspect"].casefold(),
        )

        if (
            key not in merged
            or route["C_R"] > merged[key]["C_R"]
        ):
            merged[key] = route

    final_routes = list(merged.values())

    accepted_routes = [
        route
        for route in final_routes
        if route["accepted"]
    ]

    refused_routes = [
        route
        for route in final_routes
        if not route["accepted"]
    ]

    answer_blocks = []

    if retrieve:
        for route in accepted_routes:
            data = retrieve_medical_info(
                route["disease"],
                route["aspect"],
            )

            route["retrieval"] = data

            if data and data.get("content"):
                block = {
                    "disease": route["disease"],
                    "aspect": route["aspect"],
                    "content": clean_retrieved_content(
                        data["content"]
                    ),
                    "source": data.get("source"),
                    "rating": data.get("rating"),
                    "references": data.get("references", []),
                    "images": data.get("images", []),
                }
                answer_blocks.append(block)

    result = {
        **direct,
        "final_routes": final_routes,
        "accepted_routes": accepted_routes,
        "refused_routes": refused_routes,
        "answer_blocks": answer_blocks,
        "safe_refusal": (
            SAFE_REFUSAL
            if not accepted_routes
            else None
        ),
    }

    if show:
        print("Original:", result["original_query"])

        if result["corrected_query"] != result["original_query"]:
            print("Corrected:", result["corrected_query"])

        print("\nAccepted routes:")
        if accepted_routes:
            for route in accepted_routes:
                recovery = " [T5 recovery]" if route.get("recovered_by_t5") else ""
                print(
                    f"  - {route['disease']} -> {route['aspect']} "
                    f"(C_R={route['C_R']:.4f}){recovery}"
                )
        else:
            print("  None")

        if refused_routes:
            print("\nRefused routes:")
            for route in refused_routes:
                print(
                    f"  - {route['disease']} -> {route['aspect']} "
                    f"(C_R={route['C_R']:.4f})"
                )

        if answer_blocks:
            print("\nRetrieved evidence:")
            for block in answer_blocks:
                print(
                    f"\n### {block['disease']} — {block['aspect']}\n"
                    f"{block['content']}\n"
                    f"Source: {block['source']} | Rating: {block['rating']}"
                )
        elif result["safe_refusal"]:
            print("\n" + result["safe_refusal"])

    return result
