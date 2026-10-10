BENCHMARK_CSV = os.getenv("BENCHMARK_CSV")

if not BENCHMARK_CSV:
    raise RuntimeError("Set BENCHMARK_CSV to the benchmark CSV file.")


def load_final_benchmark(path=BENCHMARK_CSV):
    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]

    last_error = None

    for encoding in encodings:
        try:
            frame = pd.read_csv(
                path,
                encoding=encoding,
            )
            break
        except Exception as exc:
            last_error = exc
    else:
        raise last_error

    required = {
        "question",
        "question Category",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Benchmark is missing required columns: {sorted(missing)}"
        )


    allowed = {
        "Straight forward",
        "Long",
        "Spelling Mistakes",
        "MedQuAD",
    }

    in_kg = frame[
        frame["question Category"]
        .astype(str)
        .str.strip()
        .isin(allowed)
    ].copy()

    category_display = {
        "Straight forward": "Straightforward",
        "Long": "Long",
        "Spelling Mistakes": "Spelling-error",
        "MedQuAD": "MedQuAD",
    }

    in_kg["Paper_Category"] = (
        in_kg["question Category"]
        .astype(str)
        .str.strip()
        .map(category_display)
    )

    return frame, in_kg
