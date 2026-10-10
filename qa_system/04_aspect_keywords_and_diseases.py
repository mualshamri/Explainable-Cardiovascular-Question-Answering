ASPECT_KEYWORDS = {
    "overview": [
        "overview", "summary",
    ],
    "symptoms": [
        "symptom", "symptoms", "sign", "signs",
    ],
    "when to see a doctor": [
        "when to see a doctor",
        "should i see a doctor",
        "when should i see a doctor",
        "see a doctor",
        "seek medical attention",
        "when to seek help",
        "when to seek medical attention",
        "when to see a gp",
        "should i go to the er",
        "when to go to the hospital",
        "emergency",
    ],
    "causes": [
        "cause", "causes", "reason", "reasons",
    ],
    "complications": [
        "complication", "complications", "risk", "risks",
    ],
    "prevention": [
        "prevention", "prevent", "preventing", "prevented",
        "avoid", "reduce", "lower",
    ],
    "diagnosis": [
        "diagnosis", "diagnose", "diagnosed",
        "test", "tests", "examination", "exam",
        "detection", "detected",
    ],
    "treatment": [
        "treatment", "treat", "treated", "cure", "management",
    ],
}


def get_all_diseases():
    query = """
    MATCH (d:Disease)
    WHERE d.name IS NOT NULL
    RETURN DISTINCT d.name AS disease
    ORDER BY toLower(d.name)
    """
    with neo4j_driver.session() as session:
        return [
            record["disease"]
            for record in session.run(query)
            if record["disease"]
        ]


diseases = get_all_diseases()
if not diseases:
    raise RuntimeError("No Disease nodes were found in Neo4j.")

print(f"Loaded {len(diseases)} disease nodes.")


disease_embeddings_cache = {
    disease: encode_all(disease)
    for disease in diseases
}

aspect_embeddings_cache = {
    aspect: encode_all(aspect)
    for aspect in ASPECT_KEYWORDS
}

print("Disease and aspect embedding caches built.")
