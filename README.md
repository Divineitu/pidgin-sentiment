# Nigerian Pidgin Tweet Sentiment

Sentiment classification (positive / negative / neutral) for tweets written in Nigerian Pidgin.

- **Live app:** https://huggingface.co/spaces/DivineItu/pidgin-sentiment-app
- **Fine-tuned model:** https://huggingface.co/DivineItu/pidgin-sentiment
- **Demo video:** _link coming_
- **Report:** _link coming_

## The problem

Nigerian Pidgin is spoken by tens of millions of people across Nigeria, and it's all over Nigerian
social media. It's an English-based creole with its own grammar ("i dey go" = "I am going"), no fixed
spelling ("dis" / "this", "dey" / "de") and lots of mixing with English, Yoruba and Igbo. Very few NLP
tools exist for it. A sentiment model for Pidgin could help businesses, NGOs or researchers understand
what Nigerians are saying online without needing everything to be in standard English.

This project builds and compares several models for Pidgin tweet sentiment, from a simple baseline up to
fine-tuned transformers, and deploys the best one as a web app.

## Dataset

[AfriSenti](https://github.com/afrisenti-semeval/afrisent-semeval-2023) Nigerian Pidgin (`pcm`) subset
(Muhammad et al., 2023), downloaded from the Hugging Face Hub (`masakhane/afrisenti`). The tweets were
collected and labelled by native speakers in the NaijaSenti project (Muhammad et al., 2022).
Licence: CC BY-NC-SA, so the raw data is **not** stored in this repo. `src/download_data.py` gets it.

| split | tweets | negative | neutral | positive |
|---|---|---|---|---|
| train (official) | 5,121 | 63.3% | 1.4% | 35.3% |
| dev (official) | 1,281 | 63.5% | 1.6% | 34.9% |
| test (official) | 4,154 | 56.0% | 10.4% | 33.6% |

### Data problems I found

1. **Leakage:** 926 of the 4,154 official test tweets (22%) are exact copies of training tweets, and 12%
   of those copies have a *different* label. A model can just remember them, so I report results on two
   test sets:
   - **official test** (all 4,154): comparable with published AfriSenti results
   - **clean test** (3,228): only tweets that never appear in train. This is the honest score.
2. **Duplicates and conflicting labels inside train:** 512 duplicates, 31 tweets with two different labels.
   Dropped (train: 5,121 → 4,577). Dev tweets that also appear in train are dropped too (1,281 → 996).
3. **Neutral is almost missing from training:** only 66 neutral tweets after cleaning, but 10% of the
   test set is neutral.
4. **Spelling variation:** e.g. dey 4,427 / de 353 / dy 28 times. 52% of the vocabulary appears only once.

### Preprocessing

Kept light on purpose, because slang and spelling carry the sentiment (`src/preprocess.py`):
lowercase, remove invisible characters left over from removed emojis, shorten 3+ repeated letters to 2
("sweeeet" → "sweet"), and normalise spaces. The app uses the same function, so the model sees the same
kind of text in training and in real use.

## Methodology

Main metric: **macro-F1** (the average of the per-class F1 scores), because the classes are imbalanced and
accuracy hides a model that ignores a whole class. Weighted F1 is also logged because the AfriSenti paper
reports it. All settings were chosen on dev; the test sets were only used for the final numbers. Every
model uses class weights in the loss (rare classes cost more when wrong). Neural models were run with 3
seeds (42, 1, 7), and I report the mean ± std.

| ID | Model | Why |
|---|---|---|
| E0 | Majority class | floor: what "no learning" gets |
| E1 | TF-IDF word 1–2 grams + logistic regression | **baseline** |
| E2 | TF-IDF character 2–5 grams + logistic regression | can letter chunks handle the spelling variation? |
| E3 | BiLSTM, random embeddings | RNN that reads word order |
| E4 | BiLSTM, Word2Vec embeddings trained on the tweets | do pretrained static embeddings help? |
| E5 | RoBERTa-base (English only) | is an English model enough for Pidgin? |
| E6 | XLM-R-base (100 languages, no Pidgin) | multilingual control for E7 |
| E7 | AfroXLMR-base (XLM-R adapted to African languages incl. Pidgin) | does African pretraining help? |
| E8 | AfriBERTa-large (African languages incl. Pidgin, from scratch) | a different African model |
| E9 | AfroXLMR-base + neutral tweets oversampled ×10 | can we fix the neutral class? |

Transformer settings: lr 2e-5, batch 32, 5 epochs, max length 64 tokens, 10% warmup, weight decay 0.01,
best epoch picked on dev macro-F1. Trained on a Colab T4 GPU.

## Results

Clean test (tweets never seen in training), mean of 3 seeds for E3–E8:

| ID | Model | Accuracy | **Macro-F1** | Weighted F1 | F1 neg | F1 neu | F1 pos |
|---|---|---|---|---|---|---|---|
| E0 | majority class | 0.529 | 0.231 | 0.366 | 0.692 | 0.000 | 0.000 |
| E1 | TF-IDF word + LR | 0.598 | 0.398 | 0.540 | 0.713 | 0.000 | 0.482 |
| E1b | TF-IDF word + LR (class weights) | 0.622 | 0.433 | 0.576 | 0.722 | 0.005 | 0.573 |
| E2 | TF-IDF char + LR | 0.605 | 0.403 | 0.546 | 0.721 | 0.000 | 0.488 |
| E2b | TF-IDF char + LR (class weights) | 0.602 | 0.427 | 0.563 | 0.698 | 0.018 | 0.566 |
| E3 | BiLSTM random emb. | 0.545 | 0.383 ± 0.002 | 0.508 | 0.644 | 0.019 | 0.487 |
| E4 | BiLSTM Word2Vec | 0.574 | 0.403 ± 0.010 | 0.535 | 0.686 | 0.025 | 0.498 |
| E5 | RoBERTa-base | 0.640 | 0.453 ± 0.002 | 0.600 | 0.731 | 0.000 | 0.629 |
| E6 | XLM-R-base | 0.640 | 0.449 ± 0.004 | 0.597 | 0.738 | 0.000 | 0.610 |
| **E7** | **AfroXLMR-base** | **0.664** | **0.464 ± 0.006** | **0.616** | **0.760** | 0.000 | **0.632** |
| E8 | AfriBERTa-large | 0.617 | 0.430 ± 0.001 | 0.573 | 0.719 | 0.003 | 0.567 |
| E9 | AfroXLMR + neutral ×10 | _pending_ | | | | | |

Scores on dev and on the official test are in `results/experiments.csv`. Every model scores higher on the
official test than on the clean test (e.g. E7: 0.485 vs 0.464), which shows the effect of the leakage.

### Sanity check against the published results

On the official test set, with the weighted F1 the AfriSenti paper uses, my numbers are close to the
published Pidgin baselines (Muhammad et al., 2023), and the ranking is the same:

| model | AfriSenti paper | mine (official test) |
|---|---|---|
| AfriBERTa-large | 62.3 | 63.8 |
| XLM-R-base | 63.8 | 65.0 |
| AfroXLMR-base | 67.6 | 66.7 |

So the pipeline behaves as expected. The clean test then shows the real scores are lower once the
duplicated tweets are removed.

### What I learned

- **Pretrained transformers beat the baseline, the BiLSTMs didn't.** With only ~4.5k tweets the BiLSTM has
  to learn word meanings from scratch and overfits. Word2Vec trained on this little text gives
  nonsense neighbours ("sweet" → "convulse", "tagged"), so it barely helps.
- **African-language pretraining helps.** AfroXLMR (E7) beats XLM-R (E6), which has the same architecture and
  size. The only difference is the extra training on African languages, Pidgin included.
- **But an English model gets most of the way.** RoBERTa (English only) came second, because most Pidgin
  words come from English.
- **"African" alone isn't enough.** AfriBERTa came last among the transformers, about the same as the baseline.
- **Neutral is never predicted** by any transformer, even with class weights, because 66 examples isn't enough.

### Error analysis (E7, clean test)

| | tweets | accuracy |
|---|---|---|
| mostly English words | 103 | 0.767 |
| light Pidgin | 687 | 0.751 |
| Pidgin | 1,468 | 0.660 |
| heavy Pidgin | 970 | 0.575 |

| tweet length | tweets | accuracy |
|---|---|---|
| 1–8 words | 734 | 0.515 |
| 9–15 | 1,010 | 0.631 |
| 16–25 | 835 | 0.717 |
| 26+ | 649 | 0.783 |

The more Pidgin a tweet has, the worse the model does, so even the best model leans on English words.
Short tweets give it too little context. Common error types: sarcasm, negative-looking words used playfully
("no use laugh kill pesin abeg" is positive), Yoruba/Igbo words mixed in, and labels that are arguable.
See `notebooks/05_error_analysis.ipynb`.

## Web app

Hugging Face only hosts Gradio Spaces on a paid plan now, but static Spaces are free, so the live app is a
single HTML page (`app/static/index.html`) that runs the model **in the browser** with
[transformers.js](https://huggingface.co/docs/transformers.js). It cleans the input the same way as
`clean_text` in training, runs the model and shows the probability of each class. Nothing typed is sent to a server.
The first visit downloads the model (~550 MB), then the browser keeps it.

To get there, `app/export_onnx.py` converts the model to ONNX. I compared three sizes on 600 clean-test tweets:

| version | size | same label as full model | accuracy |
|---|---|---|---|
| fp32 (full) | 1112 MB | 100% | 0.650 |
| int8 | 279 MB | 88.5% | 0.632 |
| **fp16 (used)** | **556 MB** | **100%** | **0.650** |

While testing I found that the browser predictions didn't match Python. transformers v5 saves the tokenizer
in a newer format that transformers.js reads wrong (words lost their start-of-word marker, so "dis" became a
different token). Using the base model's tokenizer file (same vocabulary, older format) fixed it. The token ids
are identical on all 3,228 clean-test tweets, and the browser now matches Python to within 0.1%.

`app/app.py` is a Gradio version for running locally with Python: `python app/app.py`.
To deploy your own copy: `hf auth login`, then `python export_onnx.py` and `python deploy_space.py` from `app/`.

## How to reproduce

Python 3.12 (gensim doesn't install on 3.14 yet).

```bash
python -m venv .venv
.venv\Scripts\activate          # windows  (source .venv/bin/activate on mac/linux)
pip install -r requirements.txt
python src/prepare_data.py       # download + clean the data
```

Then run the notebooks in order:

| notebook | what it does | where |
|---|---|---|
| `01_data_and_eda.ipynb` | load, explore, find the data problems, clean | CPU |
| `02_baselines.ipynb` | E0–E2 | CPU, ~1 min |
| `03_bilstm.ipynb` | E3–E4 | CPU, ~10 min |
| `04_transformers.ipynb` | E5–E8, saves the best model and pushes it to the Hub | GPU (Colab T4), ~40 min |
| `05_error_analysis.ipynb` | error analysis of the best model | CPU |
| `06_neutral_oversampling.ipynb` | E9 | GPU, ~10 min |

Run the app locally: `python app/app.py` (Gradio), or serve `app/static/` with `python -m http.server`

## Repo structure

```
src/            data download, cleaning, metrics + experiment log
notebooks/      01-06, the full workflow
results/        experiments.csv (every run), figures, predictions used in the error analysis
app/            browser app (static/index.html), onnx export, space deploy script, local gradio version
```

`results/` contains some tweets (predictions for error analysis). These come from AfriSenti and fall under
the same CC BY-NC-SA licence.

## What I used vs what I did

**Used (existing resources):** the AfriSenti dataset; pretrained models RoBERTa, XLM-R, AfroXLMR and
AfriBERTa from the Hugging Face Hub; libraries scikit-learn, PyTorch, gensim, Hugging Face transformers
and datasets, Gradio, ONNX Runtime and transformers.js.

**Did myself:** found and handled the train/test leakage and label conflicts, the cleaning pipeline, all
baselines, the BiLSTM model and training loop, Word2Vec training, the fine-tuning setup with weighted loss,
the experiment design (10 experiments, 3 seeds each for the neural ones), evaluation on the official and clean
test sets, the error analysis, and the deployed app.

## References

- Muhammad, S. H., et al. (2023). AfriSenti: A Twitter sentiment analysis benchmark for African languages. *EMNLP 2023*.
- Muhammad, S. H., et al. (2022). NaijaSenti: A Nigerian Twitter sentiment corpus for multilingual sentiment analysis. *LREC 2022*.
- Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. *COLING 2022*.
- Ogueji, K., Zhu, Y., & Lin, J. (2021). Small data? No problem! Exploring the viability of pretrained multilingual language models for low-resourced languages. *MRL Workshop 2021*.
- Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning at scale. *ACL 2020*.
- Liu, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv:1907.11692*.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*.
- Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space. *arXiv:1301.3781*.
