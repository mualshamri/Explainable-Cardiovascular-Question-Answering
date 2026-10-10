question = "What are the symptoms of cardiomyopathy?"

result = answer_question(
    question,
    retrieve=True,
    show=True,
)

if result["accepted_routes"]:
    ctra_result = ctra_explain_answer(result)
    display(display_ctra_table(ctra_result))

    print(
        "\nCTRA displayed Impact Weights sum to:",
        round(ctra_result["weight_sum"], 6),
        "%",
    )
