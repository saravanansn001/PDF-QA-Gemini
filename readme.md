# AIChatBot Backend (Python)

This folder contains a small Python backend that supports two workflows:

- **`app.py`**: PDF → chunking → embeddings (Gemini) → **FAISS** vector index → Q&A (Gemini)
- **`checkGemini.py`**: Simple **Flask** API (`POST /chat`) that calls Gemini and returns a reply

## Prerequisites

- **Python 3.9+**
- A Gemini API key in an environment variable named **`GEMINI_API_KEY`**

Create a `.env` file in this folder:

```bash
cd /path/to/your/project/backend
cat > .env <<'EOF'
GEMINI_API_KEY=YOUR_KEY_HERE
EOF
```

## Install dependencies

You have two choices:

### Option A (recommended on macOS Apple Silicon): Conda / Miniforge (best for FAISS)

FAISS often fails to install via `pip` on Apple Silicon, so use conda for FAISS.

1) Create and activate a conda env:

```bash
conda create -n aichatbot python=3.9 -y
conda activate aichatbot
```

2) Install Python dependencies:

```bash
cd /path/to/your/project/backend
pip install -r requirements.txt
```

### Option B: `venv` + pip (may fail for FAISS on Apple Silicon)
Add faiss prebuilt version in requirements.txt

```bash
cd /path/to/your/project/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you need FAISS and pip fails, switch to **Option A** above.

## Run

### Run the PDF Q&A pipeline (`app.py`)

1) Open `app.py` and update:
- `pdf_path` (currently hardcoded)
- the question passed into `get_response(...)`

2) Run:

```bash
cd /path/to/your/project/backend
python app.py
```

What it does:
- Reads the PDF
- Splits into chunks
- Builds a FAISS index and saves it under `vector_store/`
- Loads the index and runs a QA chain
- Prints the answer to the console

### Run the Flask chat API (`checkGemini.py`)

```bash
cd /path/to/your/project/backend
python checkGemini.py
```

Then call it:

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello Gemini!"}'
```

## Troubleshooting

### FAISS install/build fails with pip

Symptoms:
- `Building wheel for faiss-cpu ... error`
- SWIG errors
- `Could not import faiss python package`

Fix:
- Use conda: `conda install -c conda-forge faiss-cpu -y`

### “allow_dangerous_deserialization” error when loading FAISS

LangChain warns because FAISS uses pickle internally.
In `app.py`, loading uses:
- `allow_dangerous_deserialization=True`

Only do this for **indexes you created locally and trust**.