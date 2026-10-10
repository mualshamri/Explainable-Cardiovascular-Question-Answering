t5_tokenizer = T5Tokenizer.from_pretrained(
    T5_MODEL_NAME,
    legacy=False,
)

t5_model = T5ForConditionalGeneration.from_pretrained(
    T5_MODEL_NAME
).to(DEVICE).eval()


def generate_t5_reformulations(question):
    set_seed(T5_RANDOM_SEED)
    prompt = (
        "paraphrase: "
        f"{question} </s>"
    )

    encoded = t5_tokenizer.encode(
        prompt,
        return_tensors="pt",
        max_length=64,
        truncation=True,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = t5_model.generate(
            encoded,
            max_length=64,
            num_beams=5,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            num_return_sequences=T5_NUM_ALTERNATIVES,
            early_stopping=True,
        )

    return [
        t5_tokenizer.decode(
            output,
            skip_special_tokens=True,
        ).strip()
        for output in outputs
    ]


SAFE_REFUSAL = (
    "Warning: We don't have a satisfactory answer for this question "
    "despite several paraphrasing attempts."
)


def recover_low_support_route(route):
    """
    Reformulate one low-support routed request exactly once.
    One T5 generation call returns three alternatives.
    """
    source_text = route["segment_routing_text"]

    candidates = generate_t5_reformulations(source_text)

    evaluated = []

    for candidate in candidates:

        candidate_result = route_question_once(candidate)

        for candidate_route in candidate_result["routes"]:
            candidate_route = dict(candidate_route)
            candidate_route["candidate_text"] = candidate
            candidate_route["candidate_full_result"] = candidate_result
            evaluated.append(candidate_route)

    if not evaluated:
        failed = dict(route)
        failed["accepted"] = False
        failed["recovery_candidates"] = candidates
        return failed

    best = max(
        evaluated,
        key=lambda item: item["C_R"],
    )

    recovered = dict(best)


    recovered["segment_index"] = route.get("segment_index")
    recovered["original_segment_text"] = route.get("segment_text")
    recovered["original_segment_routing_text"] = route.get("segment_routing_text")

    recovered["recovered_by_t5"] = True
    recovered["recovery_candidates"] = candidates
    recovered["accepted"] = bool(
        recovered["C_R"] >= ROUTING_THRESHOLD
    )

    return recovered
