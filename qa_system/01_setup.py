import os
import re
import math
import time
import difflib
import warnings
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import spacy

from neo4j import GraphDatabase
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModel,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    set_seed,
)

warnings.filterwarnings("ignore")


ALPHA = 0.50
BETA = 1.50
EPS = 1e-9


ENCODER_PRIORS = {
    "biomedbert": 1.0 / 3.0,
    "biobert": 1.0 / 3.0,
    "mpnet": 1.0 / 3.0,
}

ROUTING_THRESHOLD = 0.925

T5_MODEL_NAME = "ramsrigouthamg/t5_paraphraser"
T5_NUM_ALTERNATIVES = 3
T5_RANDOM_SEED = 42

LIME_NUM_SAMPLES = 300
LIME_RANDOM_STATE = 42
LIME_MAX_FEATURES = 12
LIME_TOP_POSITIVE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_md' is required. Install it with: "
        "python -m spacy download en_core_web_md"
    )

print("Device:", DEVICE)
print("Equal encoder priors:", ENCODER_PRIORS)
print("Routing threshold:", ROUTING_THRESHOLD)
