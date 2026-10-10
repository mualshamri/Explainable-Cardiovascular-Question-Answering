def final_consistency_audit(questions, categories=None):
    if categories is None:
        categories = ["Unspecified"] * len(questions)

    rows = []
    alternative_rows = []

    for question, category in zip(questions, categories):
        probe = route_preserving_alternatives(question)

        generated = probe["generated"]
        retained = probe["retained"]

        ctra_score = np.nan
        lime_score = np.nan

        if probe["eligible"]:
            method_scores = {}

            for method in ["CTRA", "LIME"]:
                explanation_sets = []

                for item in retained:
                    words = []

                    for route in item["accepted_routes"]:
                        if method == "CTRA":
                            words.extend(
                                ctra_selected_words_for_evaluation(route)
                            )
                        else:
                            words.extend(
                                lime_selected_words(
                                    route["accepted_text"],
                                    route["disease"],
                                    route["aspect"],
                                )
                            )

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

                method_scores[method] = (
                    float(np.mean(pair_scores))
                    if pair_scores
                    else np.nan
                )

            ctra_score = method_scores["CTRA"]
            lime_score = method_scores["LIME"]

        duplicate_count = sum(
            item["status"] == "duplicate"
            for item in probe["audit"]
        )

        excluded_count = sum(
            item["status"] in {
                "route_changed",
                "below_threshold_or_incomplete",
            }
            for item in probe["audit"]
        )

        rows.append({
            "Question": question,
            "Category": category,
            "Generated": len(generated),
            "Exact_Duplicates": duplicate_count,
            "Route_Preserving": len(retained),
            "Route_Changed_or_Below_Threshold": excluded_count,
            "Eligible": int(probe["eligible"]),
            "CTRA_Consistency": ctra_score,
            "LIME_Consistency": lime_score,
        })

        for item in probe["audit"]:
            alternative_rows.append({
                "Question": question,
                "Category": category,
                **item,
            })

    question_audit = pd.DataFrame(rows)
    alternative_audit = pd.DataFrame(alternative_rows)

    category_summary = (
        question_audit
        .groupby("Category", as_index=False)
        .agg(
            Questions=("Question", "count"),
            Generated=("Generated", "sum"),
            Exact_Duplicates=("Exact_Duplicates", "sum"),
            Route_Preserving=("Route_Preserving", "sum"),
            Excluded=(
                "Route_Changed_or_Below_Threshold",
                "sum",
            ),
            Eligible_Questions=("Eligible", "sum"),
            CTRA_Consistency=(
                "CTRA_Consistency",
                lambda s: 100.0 * s.dropna().mean(),
            ),
            LIME_Consistency=(
                "LIME_Consistency",
                lambda s: 100.0 * s.dropna().mean(),
            ),
        )
    )

    total_generated = int(question_audit["Generated"].sum())
    total_retained = int(question_audit["Route_Preserving"].sum())
    total_eligible = int(question_audit["Eligible"].sum())
    total_insufficient = int(
        len(question_audit) - total_eligible
    )

    retained_pct = (
        100.0 * total_retained / total_generated
        if total_generated
        else np.nan
    )

    print("Total questions:", len(question_audit))
    print("Total alternatives generated:", total_generated)
    print("Route-preserving alternatives:", total_retained)
    print(f"Route-preserving percentage: {retained_pct:.2f}%")
    print("Questions eligible for consistency:", total_eligible)
    print(
        "Questions with fewer than two retained alternatives:",
        total_insufficient,
    )

    return (
        category_summary,
        question_audit,
        alternative_audit,
    )
