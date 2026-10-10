BIOMEDBERT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.1"
MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"

biomedbert_tokenizer = AutoTokenizer.from_pretrained(BIOMEDBERT_MODEL)
biomedbert_model = AutoModel.from_pretrained(BIOMEDBERT_MODEL).to(DEVICE).eval()

biobert_tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
biobert_model = AutoModel.from_pretrained(BIOBERT_MODEL).to(DEVICE).eval()

mpnet_model = SentenceTransformer(MPNET_MODEL, device=str(DEVICE))


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


def _hf_embedding(text, tokenizer, model, max_length=512):
    encoded = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)
        pooled = _mean_pool(
            output.last_hidden_state,
            encoded["attention_mask"],
        )

    return pooled.squeeze(0).detach().cpu().numpy().astype(np.float32)


def get_biomedbert_embedding(text):
    return _hf_embedding(text, biomedbert_tokenizer, biomedbert_model)


def get_biobert_embedding(text):
    return _hf_embedding(text, biobert_tokenizer, biobert_model)


def get_mpnet_embedding(text):
    return np.asarray(
        mpnet_model.encode(
            str(text),
            convert_to_numpy=True,
            normalize_embeddings=False,
        ),
        dtype=np.float32,
    )


def encode_all(text):
    return {
        "biomedbert": get_biomedbert_embedding(text),
        "biobert": get_biobert_embedding(text),
        "mpnet": get_mpnet_embedding(text),
    }


print("Three frozen encoders loaded.")
