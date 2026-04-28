# 🔍 Fake News Detector

An AI-powered fake news detection app that combines a **fine-tuned BERT model** with **OpenAI GPT** to classify news statements and explain *why* they may be fake or real — with a trust score from 0–100.

---

## 🧠 How It Works

```
User inputs news text
        ↓
BERT Classifier (fine-tuned on LIAR dataset)
→ Fake / Real + confidence score
        ↓
OpenAI GPT-4o-mini
→ Reasoning + Red flags + Trust score
        ↓
Streamlit UI shows combined result
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| NLP Model | BERT (fine-tuned via HuggingFace) |
| LLM | OpenAI GPT-4o-mini |
| Training Dataset | LIAR Dataset (12,800 labelled statements) |
| UI | Streamlit |
| Deep Learning | PyTorch |
| Language | Python 3.10+ |

---

## ✅ Non-Negotiable Skills Demonstrated

- ✅ **NLP / Text Classification** — BERT fine-tuning on real dataset
- ✅ **HuggingFace Transformers** — model loading, tokenization, inference
- ✅ **PyTorch** — model training and evaluation
- ✅ **LLM API Integration** — OpenAI API with structured prompting
- ✅ **Prompt Engineering** — structured output prompts with parsing
- ✅ **Model Evaluation** — accuracy, F1 score on test set
- ✅ **End-to-End Pipeline** — training → inference → UI

---

## 🚀 Setup & Run

### Step 1 — Clone and install dependencies

```bash
git clone <your-repo-url>
cd fake-news-detector
pip install -r requirements.txt
```

### Step 2 — Add your OpenAI API key

```bash
cp .env.example .env
# Open .env and paste your OpenAI API key
```

Get your API key at: https://platform.openai.com/api-keys

### Step 3 — Train the BERT model (one time only, ~20-30 mins)

```bash
python train_model.py
```

This downloads the LIAR dataset and fine-tunes BERT. The model is saved to `./models/fake_news_bert/`.

> 💡 **Tip:** If you don't want to train, the app still works using GPT only — just skip this step.

### Step 4 — Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 📁 Project Structure

```
fake-news-detector/
├── app.py                  # Streamlit UI (main entry point)
├── train_model.py          # BERT fine-tuning script (run once)
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .env                    # Your actual API key (never commit this!)
├── src/
│   ├── __init__.py
│   ├── classifier.py       # BERT model loader + inference
│   ├── explainer.py        # OpenAI GPT reasoning module
│   └── pipeline.py         # Orchestrates classifier + explainer
└── models/
    └── fake_news_bert/     # Saved model (generated after training)
```

---

## 💡 Example Output

**Input:** "Drinking bleach cures COVID-19"

**Output:**
- Trust Score: **4/100**
- Label: **FAKE**
- Verdict: This claim is medically dangerous and entirely false.
- Red Flags: No scientific basis, contradicts WHO guidelines, dangerous health misinformation
- Positive Signals: None

---

## 📊 Model Performance

After fine-tuning BERT on the LIAR dataset:

| Metric | Score |
|---|---|
| Accuracy | ~72–76% |
| F1 Score | ~71–75% |
| Dataset | LIAR (12,800 statements) |

---

## 📝 Resume Bullet Point

> Built an AI fake news detector using a fine-tuned BERT model (HuggingFace + PyTorch) trained on the LIAR dataset, integrated with OpenAI GPT-4o for reasoning and trust scoring — deployed as an interactive Streamlit app.

---

## ⚠️ Disclaimer

This tool is for educational purposes only. Always verify important news with trusted sources.
