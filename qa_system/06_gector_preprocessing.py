try:
    from gector import GECToR, predict
except Exception as exc:
    raise ImportError(
        "GECToR is required because the paper uses GECToR for preprocessing. "
        "Install the same GECToR package/environment used for the experiment. "
        "No T5 grammar fallback is enabled."
    ) from exc

GECTOR_MODEL_NAME = "gotutiyan/gector-roberta-base-5k"

_gector_model = GECToR.from_pretrained(GECTOR_MODEL_NAME)
_gector_tokenizer = AutoTokenizer.from_pretrained(GECTOR_MODEL_NAME)


def _gector_correct(text):
    result = predict(
        _gector_model,
        _gector_tokenizer,
        [str(text)],
        None,
        n_iteration=5,
    )
    return result[0] if result else str(text)


def _protect_disease_names(text):
    protected = str(text)
    restore = {}

    for index, disease in enumerate(
        sorted(diseases, key=lambda x: len(x), reverse=True)
    ):
        pattern = re.compile(
            rf"(?<!\w){re.escape(disease)}(?!\w)",
            re.IGNORECASE,
        )

        if pattern.search(protected):
            placeholder = f"ZZDISEASE{index}ZZ"
            protected = pattern.sub(placeholder, protected)
            restore[placeholder] = disease

    return protected, restore


def _restore_disease_names(text, restore):
    restored = str(text)
    for placeholder, disease in restore.items():
        restored = re.sub(
            re.escape(placeholder),
            disease,
            restored,
            flags=re.IGNORECASE,
        )
    return restored


def _surface_key(text):
    return re.sub(r"\s+", "", str(text)).casefold()


def _edit_diff(original, corrected):
    original_tokens = str(original).split()
    corrected_tokens = str(corrected).split()

    matcher = difflib.SequenceMatcher(
        a=original_tokens,
        b=corrected_tokens,
    )

    edits = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        before = " ".join(original_tokens[i1:i2])
        after = " ".join(corrected_tokens[j1:j2])

        if _surface_key(before) == _surface_key(after):
            continue

        edits.append({
            "type": tag,
            "from": before,
            "to": after,
        })

    return edits


def grammar_correct(text):
    protected, restore = _protect_disease_names(text)
    corrected = _gector_correct(protected)
    corrected = _restore_disease_names(corrected, restore)

    return {
        "original": str(text),
        "corrected": corrected,
        "edits": _edit_diff(text, corrected),
        "shielded": list(restore.values()),
        "backend": "GECToR",
    }


print("GECToR preprocessing ready. No T5 grammar fallback is enabled.")
