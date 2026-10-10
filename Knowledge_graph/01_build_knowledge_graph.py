import ast
import os
import getpass
import hashlib
import re
import pandas as pd
from neo4j import GraphDatabase

csv_file_path = os.getenv("KG_CSV_PATH")

if not csv_file_path:
    raise RuntimeError(
        "Set KG_CSV_PATH to the cardiovascular dataset CSV file."
    )

NEO4J_URI = os.getenv(
    "NEO4J_URI",
    "bolt://localhost:7687"
)

NEO4J_USER = os.getenv(
    "NEO4J_USER",
    "neo4j"
)

NEO4J_PASSWORD = os.getenv(
    "NEO4J_PASSWORD"
)

if not NEO4J_PASSWORD:
    NEO4J_PASSWORD = getpass.getpass(
        "Enter Neo4j password: "
    )

REBUILD_GRAPH = True

NO_IMAGE_PLACEHOLDER = "there is no relevant image"

ASPECT_RELATIONSHIPS = {
    "overview": (
        "Overview",
        "HAS_OVERVIEW"
    ),
    "symptoms": (
        "Symptoms",
        "HAS_SYMPTOMS"
    ),
    "when to see a doctor": (
        "WhenToSeeDoctor",
        "HAS_WHENTOSEEDOCTOR"
    ),
    "causes": (
        "Causes",
        "HAS_CAUSES"
    ),
    "complications": (
        "Complications",
        "HAS_COMPLICATIONS"
    ),
    "prevention": (
        "Prevention",
        "HAS_PREVENTION"
    ),
    "diagnosis": (
        "Diagnosis",
        "HAS_DIAGNOSIS"
    ),
    "treatment": (
        "Treatment",
        "HAS_TREATMENT"
    ),
}

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(
        NEO4J_USER,
        NEO4J_PASSWORD
    )
)


def clean_text(value):

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if text == "":
        return None

    return text


def is_missing_image(value):

    value = clean_text(value)

    if value is None:
        return True

    return (
        value.lower()
        == NO_IMAGE_PLACEHOLDER.lower()
    )


def stable_id(*parts):

    text = "||".join(
        ""
        if part is None
        else str(part).strip().lower()
        for part in parts
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def parse_references(raw):

    raw = clean_text(raw)

    if raw is None:
        return []

    references = []

    try:

        parsed = ast.literal_eval(raw)

        if isinstance(
            parsed,
            (list, tuple)
        ):

            references = [
                clean_text(x)
                for x in parsed
            ]

        elif isinstance(
            parsed,
            str
        ):

            references = [
                clean_text(parsed)
            ]

        else:

            references = [
                clean_text(parsed)
            ]

    except (ValueError, SyntaxError):

        references = [
            clean_text(x)
            for x in raw.split(";")
        ]

    references = [
        x
        for x in references
        if x is not None
    ]

    return list(
        dict.fromkeys(references)
    )


def clear_database(session):

    if not REBUILD_GRAPH:
        return

    session.run(
        """
        MATCH (n)
        DETACH DELETE n
        """
    )


def create_constraints(session):

    constraints = [

        """
        CREATE CONSTRAINT disease_unique IF NOT EXISTS
        FOR (n:Disease)
        REQUIRE n.name IS UNIQUE
        """,

        """
        CREATE CONSTRAINT metadata_unique IF NOT EXISTS
        FOR (n:Metadata)
        REQUIRE n.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT overview_unique IF NOT EXISTS
        FOR (n:Overview)
        REQUIRE n.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT symptoms_unique IF NOT EXISTS
        FOR (n:Symptoms)
        REQUIRE n.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT when_to_see_doctor_unique IF NOT EXISTS
        FOR (n:WhenToSeeDoctor)
        REQUIRE n.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT causes_unique IF NOT EXISTS
        FOR (n:Causes)
        REQUIRE n.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT complications_unique IF NOT EXISTS
        FOR (n:Complications)
        REQUIRE n.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT prevention_unique IF NOT EXISTS
        FOR (n:Prevention)
        REQUIRE n.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT diagnosis_unique IF NOT EXISTS
        FOR (n:Diagnosis)
        REQUIRE n.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT treatment_unique IF NOT EXISTS
        FOR (n:Treatment)
        REQUIRE n.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT source_unique IF NOT EXISTS
        FOR (n:Source)
        REQUIRE n.name IS UNIQUE
        """,

        """
        CREATE CONSTRAINT source_rating_unique IF NOT EXISTS
        FOR (n:SourceRating)
        REQUIRE n.value IS UNIQUE
        """,

        """
        CREATE CONSTRAINT reference_unique IF NOT EXISTS
        FOR (n:Reference)
        REQUIRE n.name IS UNIQUE
        """,

        """
        CREATE CONSTRAINT image_unique IF NOT EXISTS
        FOR (n:Image)
        REQUIRE n.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT image_url_unique IF NOT EXISTS
        FOR (n:ImageURL)
        REQUIRE n.url IS UNIQUE
        """,

        """
        CREATE CONSTRAINT image_source_unique IF NOT EXISTS
        FOR (n:ImageSource)
        REQUIRE n.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT image_creator_unique IF NOT EXISTS
        FOR (n:ImageCreator)
        REQUIRE n.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT image_license_unique IF NOT EXISTS
        FOR (n:ImageLicense)
        REQUIRE n.id IS UNIQUE
        """
    ]

    for query in constraints:
        session.run(query)


def write_row(
    tx,
    *,
    disease_name,
    aspect_name,
    content,
    source,
    rating,
    date,
    references,
    image_url,
    image_source,
    image_creator,
    image_license
):

    disease_name = clean_text(
        disease_name
    )

    aspect_name = clean_text(
        aspect_name
    )

    content = clean_text(
        content
    )

    source = clean_text(
        source
    )

    rating = clean_text(
        rating
    )

    date = clean_text(
        date
    )

    if disease_name is None:
        return

    if aspect_name is None:
        return

    aspect_name = aspect_name.lower()

    tx.run(
        """
        MERGE (
            d:Disease {
                name: $disease
            }
        )
        """,
        disease=disease_name
    )

    aspect_config = (
        ASPECT_RELATIONSHIPS.get(
            aspect_name
        )
    )

    if (
        aspect_config is not None
        and content is not None
    ):

        aspect_label, relationship = (
            aspect_config
        )

        aspect_id = stable_id(
            "clinical_aspect",
            disease_name,
            aspect_name
        )

        query = f"""
        MATCH (
            d:Disease {{
                name: $disease
            }}
        )

        MERGE (
            a:{aspect_label} {{
                id: $aspect_id
            }}
        )

        SET
            a.content = $content,
            a.date = $date

        MERGE (
            d
        )-[:{relationship}]->(
            a
        )
        """

        tx.run(
            query,
            disease=disease_name,
            aspect_id=aspect_id,
            content=content,
            date=date
        )

    metadata_id = stable_id(
        "metadata",
        disease_name
    )

    tx.run(
        """
        MATCH (
            d:Disease {
                name: $disease
            }
        )

        MERGE (
            m:Metadata {
                id: $metadata_id
            }
        )

        MERGE (
            d
        )-[:HAS_METADATA]->(
            m
        )
        """,
        disease=disease_name,
        metadata_id=metadata_id
    )

    if source is not None:

        tx.run(
            """
            MERGE (
                s:Source {
                    name: $source
                }
            )

            WITH s

            MATCH (
                m:Metadata {
                    id: $metadata_id
                }
            )

            MERGE (
                m
            )-[:HAS_SOURCE]->(
                s
            )
            """,
            source=source,
            metadata_id=metadata_id
        )

        if rating is not None:

            tx.run(
                """
                MERGE (
                    r:SourceRating {
                        value: $rating
                    }
                )

                WITH r

                MATCH (
                    s:Source {
                        name: $source
                    }
                )

                MERGE (
                    s
                )-[:HAS_RATING]->(
                    r
                )
                """,
                rating=rating,
                source=source
            )

    parsed_references = (
        parse_references(
            references
        )
    )

    for reference in parsed_references:

        tx.run(
            """
            MERGE (
                r:Reference {
                    name: $reference
                }
            )

            WITH r

            MATCH (
                m:Metadata {
                    id: $metadata_id
                }
            )

            MERGE (
                m
            )-[:HAS_REFERENCE]->(
                r
            )
            """,
            reference=reference,
            metadata_id=metadata_id
        )

    if not is_missing_image(
        image_url
    ):

        image_url = clean_text(
            image_url
        )

        image_source = clean_text(
            image_source
        )

        image_creator = clean_text(
            image_creator
        )

        image_license = clean_text(
            image_license
        )

        image_id = stable_id(
            "image",
            image_url
        )

        tx.run(
            """
            MERGE (
                img:Image {
                    id: $image_id
                }
            )

            WITH img

            MATCH (
                m:Metadata {
                    id: $metadata_id
                }
            )

            MERGE (
                m
            )-[:HAS_IMAGE]->(
                img
            )
            """,
            image_id=image_id,
            metadata_id=metadata_id
        )

        tx.run(
            """
            MERGE (
                u:ImageURL {
                    url: $image_url
                }
            )

            WITH u

            MATCH (
                img:Image {
                    id: $image_id
                }
            )

            MERGE (
                img
            )-[:HAS_URL]->(
                u
            )
            """,
            image_url=image_url,
            image_id=image_id
        )

        if image_source is not None:

            image_source_id = stable_id(
                "image_source",
                image_id,
                image_source
            )

            tx.run(
                """
                MERGE (
                    s:ImageSource {
                        id: $source_id
                    }
                )

                SET
                    s.name = $source_name

                WITH s

                MATCH (
                    img:Image {
                        id: $image_id
                    }
                )

                MERGE (
                    img
                )-[:HAS_IMAGE_SOURCE]->(
                    s
                )
                """,
                source_id=image_source_id,
                source_name=image_source,
                image_id=image_id
            )

        if image_creator is not None:

            creator_id = stable_id(
                "image_creator",
                image_id,
                image_creator
            )

            tx.run(
                """
                MERGE (
                    c:ImageCreator {
                        id: $creator_id
                    }
                )

                SET
                    c.name = $creator_name

                WITH c

                MATCH (
                    img:Image {
                        id: $image_id
                    }
                )

                MERGE (
                    img
                )-[:CREATED_BY]->(
                    c
                )
                """,
                creator_id=creator_id,
                creator_name=image_creator,
                image_id=image_id
            )

        if image_license is not None:

            license_id = stable_id(
                "image_license",
                image_id,
                image_license
            )

            tx.run(
                """
                MERGE (
                    l:ImageLicense {
                        id: $license_id
                    }
                )

                SET
                    l.name = $license_name

                WITH l

                MATCH (
                    img:Image {
                        id: $image_id
                    }
                )

                MERGE (
                    img
                )-[:HAS_LICENSE]->(
                    l
                )
                """,
                license_id=license_id,
                license_name=image_license,
                image_id=image_id
            )


def populate_neo4j(df):

    required_columns = {
        "disease_name",
        "aspect",
        "content",
        "source",
        "source_rating",
        "date",
        "references",
        "Image URLs",
        "image_source",
        "image_creator",
        "image_license",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing columns: "
            + str(
                sorted(
                    missing_columns
                )
            )
        )

    with driver.session() as session:

        clear_database(
            session
        )

        create_constraints(
            session
        )

        total_rows = len(df)

        for index, row in df.iterrows():

            session.execute_write(
                write_row,

                disease_name=(
                    row["disease_name"]
                ),

                aspect_name=(
                    row["aspect"]
                ),

                content=(
                    row["content"]
                ),

                source=(
                    row["source"]
                ),

                rating=(
                    row["source_rating"]
                ),

                date=(
                    row["date"]
                ),

                references=(
                    row["references"]
                ),

                image_url=(
                    row["Image URLs"]
                ),

                image_source=(
                    row["image_source"]
                ),

                image_creator=(
                    row["image_creator"]
                ),

                image_license=(
                    row["image_license"]
                ),
            )

            current = index + 1

            if (
                current % 50 == 0
                or current == total_rows
            ):

                print(
                    f"Imported "
                    f"{current}/"
                    f"{total_rows} rows"
                )


def print_node_counts(session):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "NODE COUNTS"
    )

    print(
        "=" * 60
    )

    query = """
    MATCH (n)

    UNWIND
        labels(n) AS label

    RETURN
        label,
        count(*) AS count

    ORDER BY
        count DESC,
        label
    """

    for record in session.run(
        query
    ):

        print(
            f"{record['label']:<25}"
            f"{record['count']}"
        )

    result = session.run(
        """
        MATCH (n)

        RETURN
            count(n) AS total
        """
    ).single()

    print(
        "\nTOTAL NODES:",
        result["total"]
    )


def print_relationship_counts(session):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "RELATIONSHIP COUNTS"
    )

    print(
        "=" * 60
    )

    query = """
    MATCH ()-[r]->()

    RETURN
        type(r) AS relationship,
        count(*) AS count

    ORDER BY
        count DESC,
        relationship
    """

    for record in session.run(
        query
    ):

        print(
            f"{record['relationship']:<35}"
            f"{record['count']}"
        )

    result = session.run(
        """
        MATCH ()-[r]->()

        RETURN
            count(r) AS total
        """
    ).single()

    print(
        "\nTOTAL RELATIONSHIPS:",
        result["total"]
    )


def verify_metadata_graph(session):

    query = """
    MATCH
        (d:Disease)
        -[:HAS_METADATA]->
        (m:Metadata)

    OPTIONAL MATCH
        (m)-[:HAS_SOURCE]->
        (s:Source)

    OPTIONAL MATCH
        (s)-[:HAS_RATING]->
        (r:SourceRating)

    OPTIONAL MATCH
        (m)-[:HAS_REFERENCE]->
        (ref:Reference)

    OPTIONAL MATCH
        (m)-[:HAS_IMAGE]->
        (img:Image)

    RETURN
        count(DISTINCT d) AS diseases,
        count(DISTINCT m) AS metadata_nodes,
        count(DISTINCT s) AS sources,
        count(DISTINCT r) AS ratings,
        count(DISTINCT ref) AS references,
        count(DISTINCT img) AS images
    """

    result = session.run(
        query
    ).single()

    print(
        "\nMETADATA GRAPH SUMMARY"
    )

    print(
        "Diseases:        ",
        result["diseases"]
    )

    print(
        "Metadata nodes:  ",
        result["metadata_nodes"]
    )

    print(
        "Sources:         ",
        result["sources"]
    )

    print(
        "Ratings:         ",
        result["ratings"]
    )

    print(
        "References:      ",
        result["references"]
    )

    print(
        "Images:          ",
        result["images"]
    )


def verify_image_graph(session):

    query = """
    MATCH
        (:Disease)
        -[:HAS_METADATA]->
        (:Metadata)
        -[:HAS_IMAGE]->
        (img:Image)

    OPTIONAL MATCH
        (img)-[:HAS_URL]->
        (url:ImageURL)

    OPTIONAL MATCH
        (img)-[:HAS_IMAGE_SOURCE]->
        (src:ImageSource)

    OPTIONAL MATCH
        (img)-[:CREATED_BY]->
        (creator:ImageCreator)

    OPTIONAL MATCH
        (img)-[:HAS_LICENSE]->
        (lic:ImageLicense)

    RETURN
        count(DISTINCT img) AS images,
        count(DISTINCT url) AS image_urls,
        count(DISTINCT src) AS image_sources,
        count(DISTINCT creator) AS image_creators,
        count(DISTINCT lic) AS image_licenses
    """

    result = session.run(
        query
    ).single()

    print(
        "\nIMAGE GRAPH SUMMARY"
    )

    print(
        "Image nodes:          ",
        result["images"]
    )

    print(
        "ImageURL nodes:       ",
        result["image_urls"]
    )

    print(
        "ImageSource nodes:    ",
        result["image_sources"]
    )

    print(
        "ImageCreator nodes:   ",
        result["image_creators"]
    )

    print(
        "ImageLicense nodes:   ",
        result["image_licenses"]
    )


def verify_image_relationships(session):

    checks = {

        "Metadata -> Image":
        """
        MATCH
            (:Metadata)
            -[r:HAS_IMAGE]->
            (:Image)

        RETURN
            count(r) AS count
        """,

        "Image -> ImageURL":
        """
        MATCH
            (:Image)
            -[r:HAS_URL]->
            (:ImageURL)

        RETURN
            count(r) AS count
        """,

        "Image -> ImageSource":
        """
        MATCH
            (:Image)
            -[r:HAS_IMAGE_SOURCE]->
            (:ImageSource)

        RETURN
            count(r) AS count
        """,

        "Image -> ImageCreator":
        """
        MATCH
            (:Image)
            -[r:CREATED_BY]->
            (:ImageCreator)

        RETURN
            count(r) AS count
        """,

        "Image -> ImageLicense":
        """
        MATCH
            (:Image)
            -[r:HAS_LICENSE]->
            (:ImageLicense)

        RETURN
            count(r) AS count
        """
    }

    print(
        "\nIMAGE RELATIONSHIP COUNTS"
    )

    for name, query in checks.items():

        result = session.run(
            query
        ).single()

        print(
            f"{name:<30}"
            f"{result['count']}"
        )


def verify_clinical_relationships(session):

    relationships = [
        "HAS_OVERVIEW",
        "HAS_SYMPTOMS",
        "HAS_WHENTOSEEDOCTOR",
        "HAS_CAUSES",
        "HAS_COMPLICATIONS",
        "HAS_PREVENTION",
        "HAS_DIAGNOSIS",
        "HAS_TREATMENT",
    ]

    print(
        "\nCLINICAL RELATIONSHIP COUNTS"
    )

    for relationship in relationships:

        query = f"""
        MATCH
            (:Disease)
            -[r:{relationship}]->
            ()

        RETURN
            count(r) AS count
        """

        result = session.run(
            query
        ).single()

        print(
            f"{relationship:<35}"
            f"{result['count']}"
        )


def print_sample_image_graph(session):

    query = """
    MATCH
        (d:Disease)
        -[:HAS_METADATA]->
        (:Metadata)
        -[:HAS_IMAGE]->
        (img:Image)

    OPTIONAL MATCH
        (img)-[:HAS_URL]->
        (url:ImageURL)

    OPTIONAL MATCH
        (img)-[:HAS_IMAGE_SOURCE]->
        (src:ImageSource)

    OPTIONAL MATCH
        (img)-[:CREATED_BY]->
        (creator:ImageCreator)

    OPTIONAL MATCH
        (img)-[:HAS_LICENSE]->
        (lic:ImageLicense)

    RETURN
        d.name AS disease,
        url.url AS image_url,
        src.name AS image_source,
        creator.name AS image_creator,
        lic.name AS image_license

    LIMIT 10
    """

    print(
        "\nSAMPLE IMAGE PROVENANCE"
    )

    for record in session.run(
        query
    ):

        print(
            "\nDisease:",
            record["disease"]
        )

        print(
            "Image URL:",
            record["image_url"]
        )

        print(
            "Image Source:",
            record["image_source"]
        )

        print(
            "Image Creator:",
            record["image_creator"]
        )

        print(
            "Image Licence:",
            record["image_license"]
        )


def print_graph_summary():

    with driver.session() as session:

        print_node_counts(
            session
        )

        print_relationship_counts(
            session
        )

        verify_metadata_graph(
            session
        )

        verify_clinical_relationships(
            session
        )

        verify_image_graph(
            session
        )

        verify_image_relationships(
            session
        )

        print_sample_image_graph(
            session
        )


def load_csv_flexible(path):

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1"
    ]

    last_error = None

    for encoding in encodings:

        try:

            print(
                f"Trying encoding: "
                f"{encoding}"
            )

            df = pd.read_csv(
                path,
                encoding=encoding
            )

            print(
                "Successfully loaded CSV "
                f"using: {encoding}"
            )

            return df

        except UnicodeDecodeError as error:

            last_error = error

            print(
                f"Failed with "
                f"{encoding}"
            )

    raise RuntimeError(
        "Could not read the CSV "
        "using the supported encodings."
    ) from last_error


def load_csv_to_neo4j():

    df = load_csv_flexible(
        csv_file_path
    )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CSV INFORMATION"
    )

    print(
        "=" * 60
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Unique diseases:",
        df["disease_name"]
        .dropna()
        .nunique()
    )

    print(
        "Unique aspect types:",
        df["aspect"]
        .dropna()
        .nunique()
    )

    valid_image_mask = (
        df["Image URLs"].notna()
        &
        (
            df["Image URLs"]
            .astype(str)
            .str.strip()
            .str.lower()
            !=
            NO_IMAGE_PLACEHOLDER.lower()
        )
    )

    valid_images = (
        df.loc[
            valid_image_mask
        ]
        .copy()
    )

    unique_images = (
        valid_images
        .drop_duplicates(
            subset=[
                "Image URLs"
            ]
        )
        .copy()
    )

    print(
        "Unique image URLs:",
        unique_images[
            "Image URLs"
        ].nunique()
    )

    print(
        "Unique images with source:",
        unique_images[
            "image_source"
        ].notna().sum()
    )

    print(
        "Unique images with creator:",
        unique_images[
            "image_creator"
        ].notna().sum()
    )

    print(
        "Unique images with licence:",
        unique_images[
            "image_license"
        ].notna().sum()
    )

    populate_neo4j(
        df
    )

    print(
        "\nKnowledge graph build complete."
    )

    print_graph_summary()


if __name__ == "__main__":

    try:

        load_csv_to_neo4j()

    finally:

        driver.close()
