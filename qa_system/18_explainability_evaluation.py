def evaluate_explainability(questions):
    query_rows = []
    consistency_rows = []

    for question in questions:
        answer = answer_question(
            question,
            retrieve=False,
            show=False,
        )

        for route in answer["accepted_routes"]:

            start = time.perf_counter()
            ctra_words = ctra_selected_words_for_evaluation(route)
            ctra_runtime = time.perf_counter() - start

            ctra_suff = explanation_sufficiency(
                route["accepted_text"],
                route["disease"],
                route["aspect"],
                ctra_words,
            )


            start = time.perf_counter()
            lime_words = lime_selected_words(
                route["accepted_text"],
                route["disease"],
                route["aspect"],
            )
            lime_runtime = time.perf_counter() - start

            lime_suff = explanation_sufficiency(
                route["accepted_text"],
                route["disease"],
                route["aspect"],
                lime_words,
            )

            query_rows.extend([
                {
                    "Question": question,
                    "Method": "Proposed CTRA",
                    "Disease": route["disease"],
                    "Aspect": route["aspect"],
                    "Sufficiency": ctra_suff,
                    "Length": len(set(map(normalise_phrase, ctra_words))),
                    "Runtime_s": ctra_runtime,
                },
                {
                    "Question": question,
                    "Method": "LIME",
                    "Disease": route["disease"],
                    "Aspect": route["aspect"],
                    "Sufficiency": lime_suff,
                    "Length": len(set(map(normalise_phrase, lime_words))),
                    "Runtime_s": lime_runtime,
                },
            ])


        ctra_consistency = explanation_consistency_for_question(
            question,
            method="CTRA",
        )


        lime_consistency = explanation_consistency_for_question(
            question,
            method="LIME",
        )

        consistency_rows.extend([
            {
                "Question": question,
                "Method": "Proposed CTRA",
                "Consistency": ctra_consistency["consistency"],
                "Generated": len(ctra_consistency["generated"]),
                "Retained": len(ctra_consistency["retained"]),
                "Eligible": int(ctra_consistency["eligible"]),
            },
            {
                "Question": question,
                "Method": "LIME",
                "Consistency": lime_consistency["consistency"],
                "Generated": len(lime_consistency["generated"]),
                "Retained": len(lime_consistency["retained"]),
                "Eligible": int(lime_consistency["eligible"]),
            },
        ])

    detail = pd.DataFrame(query_rows)
    consistency_detail = pd.DataFrame(consistency_rows)

    summary = (
        detail
        .groupby("Method", as_index=False)
        .agg(
            Sufficiency=("Sufficiency", lambda s: 100.0 * s.mean()),
            Avg_Length=("Length", "mean"),
            Avg_Runtime_s=("Runtime_s", "mean"),
        )
    )

    consistency_summary = (
        consistency_detail[
            consistency_detail["Eligible"].eq(1)
        ]
        .groupby("Method", as_index=False)
        .agg(
            Consistency=(
                "Consistency",
                lambda s: 100.0 * s.dropna().mean(),
            ),
            Eligible_Questions=("Question", "nunique"),
        )
    )

    summary = summary.merge(
        consistency_summary,
        on="Method",
        how="left",
    )

    return summary, detail, consistency_detail
