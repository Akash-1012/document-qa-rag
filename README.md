# Document Q&A — Retrieval-Augmented Answers

A small Streamlit app that answers questions from user-provided documents by retrieving relevant passages and showing their sources. It is designed to avoid unsupported answers: if the best passage is below a conservative similarity threshold, the app says the information was not found.

## Features

- Upload `.txt` and `.md` documents, or place them in `documents/`.
- Split documents into overlapping chunks.
- Create semantic embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Retrieve the most relevant chunks using cosine similarity.
- Show the source filename, chunk number, and similarity score beside every retrieved passage.
- Return an explicit “Not found in the provided documents” message when retrieval confidence is low.

## Requirements

- Python 3.10 or newer
- Internet access the first time the embedding model is downloaded

## Run locally

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

## How to use

1. Upload one or more `.txt` or `.md` files in the sidebar, or put files in `documents/`.
2. Enter a question based on those documents.
3. Select how many passages to retrieve and click **Find answer**.
4. Review the cited source passages. If no passage is relevant enough, the app says the answer was not found.

## Design decisions and limitations

- This is intentionally a small, transparent RAG baseline.
- The app displays retrieved evidence rather than using a second generative model to synthesize prose. This reduces hallucination risk and makes source grounding easy to inspect.
- The similarity threshold (`0.25`) is a simple heuristic, not a calibrated confidence probability. Tune it using representative documents and questions.
- Uploaded files are processed in memory for the current session and are not saved by the app.
- Only `.txt` and `.md` files are supported in this version.

## Suggested test

Create a text file containing: `The sample is stored at 4 degrees Celsius.` Ask: `At what temperature is the sample stored?` The relevant passage should appear with its filename. Then ask about something absent from the file and check the “Not found” response.

## Project structure

```text
document-qa-rag/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── documents/
    └── README.md
```
