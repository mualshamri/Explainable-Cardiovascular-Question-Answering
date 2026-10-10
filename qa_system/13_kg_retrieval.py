ASPECT_RELATIONSHIPS = {
    "overview": "HAS_OVERVIEW",
    "symptoms": "HAS_SYMPTOMS",
    "when to see a doctor": "HAS_WHENTOSEEDOCTOR",
    "causes": "HAS_CAUSES",
    "complications": "HAS_COMPLICATIONS",
    "prevention": "HAS_PREVENTION",
    "diagnosis": "HAS_DIAGNOSIS",
    "treatment": "HAS_TREATMENT",
}


def retrieve_medical_info(disease, aspect):
    relationship = ASPECT_RELATIONSHIPS.get(
        str(aspect).casefold()
    )

    if relationship is None:
        return None


    query = f"""
    MATCH (d:Disease)
    WHERE toLower(d.name) = toLower($disease)

    MATCH (d)-[:{relationship}]->(a)

    OPTIONAL MATCH (a)-[:HAS_CONTENT]->(c)
    OPTIONAL MATCH (a)-[:HAS_SOURCE]->(s)
    OPTIONAL MATCH (a)-[:HAS_RATING]->(sr)
    OPTIONAL MATCH (d)-[:HAS_REFERENCE]->(r)
    OPTIONAL MATCH (d)-[:HAS_IMAGE]->(img)

    RETURN
        coalesce(c.content, c.text, a.content) AS content,
        s.name AS source,
        coalesce(sr.value, s.rating, a.rating) AS rating,
        collect(DISTINCT r.name) AS references,
        collect(DISTINCT img.url) AS images
    """

    with neo4j_driver.session() as session:
        record = session.run(
            query,
            disease=str(disease),
        ).single()

    if record is None:
        return None

    return {
        "content": record["content"],
        "source": record["source"],
        "rating": record["rating"],
        "references": [
            value
            for value in (record["references"] or [])
            if value
        ],
        "images": [
            value
            for value in (record["images"] or [])
            if value
        ],
    }


def clean_retrieved_content(text):
    if text is None:
        return ""

    doc = nlp(str(text))

    sentences = [
        sentence.text.strip()
        for sentence in doc.sents
        if sentence.text.strip()
    ]

    unique = list(dict.fromkeys(sentences))

    kept = [
        sentence
        for sentence in unique
        if not any(
            sentence != other and sentence in other
            for other in unique
        )
    ]

    return re.sub(
        r"\s+",
        " ",
        " ".join(kept),
    ).strip()
