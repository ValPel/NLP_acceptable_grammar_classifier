# BERT Grammar Classification

## Load & Balance Data
- Load CoLA dataset from `./cola_public/raw/in_domain_train.tsv`
- Keep `labels` and `sentence` columns
- Balance to **3,000 samples** (1,500 per class)
- Split into **train/test** and save as:
  - `./data/train.csv`
  - `./data/test.csv`

---

## Tokenization & Datasets
- Use tokenizer: **`bert-base-cased`**
- Tokenize with padding and truncation
- Convert to Hugging Face `Dataset`
- Remove extra columns
- Create DataLoaders:
  - Train: batch size 16, shuffled  
  - Eval: batch size 32, not shuffled

---

## Model
- Base: `AutoModelForSequenceClassification("bert-base-cased", num_labels=2)`
- Custom classifier head: Linear(768 → 128) → ReLU → Linear(128 → 2) → LogSoftmax

---

## Optimization & Loss
- Optimizer: **AdamW**, weight decay `1e-2`
- Learning rates:
- BERT: `2e-5`
- Classifier: `1e-3`
- Loss: **NLLLoss**

---

## Training
- Runs for **2 epochs** on GPU if available
- Each step:
- Zero gradients  
- Forward pass → compute loss → backward pass  
- Clip gradients (1.0) and update weights
- Tracks loss and accuracy per epoch

---

## Evaluation
- Use `model.eval()` with `torch.no_grad()`
- Compute test accuracy and predictions

---

## Reporting
- Plot confusion matrix with:
- **Ungrammatical (0)**
- **Grammatical (1)**
