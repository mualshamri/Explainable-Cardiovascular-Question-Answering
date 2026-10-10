# Explainable Cardiovascular Question Answering

## Explainable Cardiovascular Question Answering with Controlled Knowledge-Graph Retrieval

This repository contains the implementation of the proposed cardiovascular question-answering framework. The system combines query preprocessing, disease and clinical-aspect routing, T5-based recovery, controlled knowledge-graph retrieval, out-of-KG query handling, and explainability using Causal Token Role Attribution (CTRA).

## Repository Structure

### knowledge\_graph

This folder contains the code used to construct the cardiovascular knowledge graph in Neo4j.

The main file is:

`01\\\_build\\\_knowledge\\\_graph.py`



### qa\_system

This folder contains the complete question-answering pipeline.

The Python files are numbered according to their execution order:

`01\\\_setup.py`  
`02\\\_neo4j\\\_connection.py`  
`03\\\_load\\\_encoders.py`  
`04\\\_aspect\\\_keywords\\\_and\\\_diseases.py`  
`05\\\_text\\\_normalisation.py`  
`06\\\_gector\\\_preprocessing.py`  
`07\\\_query\\\_segmentation.py`  
`08\\\_semantic\\\_scoring.py`  
`09\\\_disease\\\_routing.py`  
`10\\\_aspect\\\_routing.py`  
`11\\\_route\\\_question.py`  
`12\\\_t5\\\_recovery.py`  
`13\\\_kg\\\_retrieval.py`  
`14\\\_answer\\\_question.py`  
`15\\\_ctra.py`  
`16\\\_lime\\\_evaluation.py`  
`17\\\_consistency.py`  
`18\\\_explainability\\\_evaluation.py`  
`19\\\_consistency\\\_audit.py`  
`20\\\_benchmark\\\_loader.py`  
`21\\\_example.py`

The file `run\\\_all.py` runs the QA pipeline files in the required order.

### zero\_shot\_llm\_prompt

This file contains the zero-shot prompt used to evaluate the language-model baselines.

The prompt instructs the models to identify and rank supported disease and clinical-aspect candidates and to return no answer when the queried condition is outside the knowledge-graph scope.

### requirements.txt

This file contains the Python packages required to run the system.

## Requirements

Python 3.10 or later is recommended.

Install the required packages using:

```bash
pip install -r requirements.txt
```

Install the required spaCy language model using:

```bash
python -m spacy download en\\\_core\\\_web\\\_md
```

A running Neo4j database is also required.

## Configuration

Passwords, dataset paths, and local computer paths are not stored in the repository.

Before running the system, set the following environment variables:

`NEO4J\\\_URI`  
`NEO4J\\\_USER`  
`NEO4J\\\_PASSWORD`  
`KG\\\_CSV\\\_PATH`  
`NEO4J\\\_URI` specifies the Neo4j database connection.

`NEO4J\\\_USER` specifies the Neo4j username.

`NEO4J\\\_PASSWORD` specifies the Neo4j password.

`KG\\\_CSV\\\_PATH` specifies the path to the cardiovascular dataset used to construct the knowledge graph.

## Running the System

### Step 1: Install the required packages

### Step 2: Start Neo4j

### Step 3: Set the environment variables

### Step 4: Build the knowledge graph

### Step 5: Run the question-answering pipeline



