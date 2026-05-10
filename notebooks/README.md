# Notebooks

# Certified Adversarial Robustness Reproduction

Reproduction of **"On the Scalability of Certified 
Adversarial Robustness with Generated Data"** 
(NeurIPS 2024) for FAST-NUCES DS/AI Assignment 2 & 3.

## Paper
- Title: On the Scalability of Certified Adversarial 
  Robustness with Generated Data
- Venue: NeurIPS 2024
- Authors: Altstidl et al.

## Results

| Model | Clean Acc | Cert Acc |
|---|---|---|
| Baseline (no aux) | 29.97% | 24.65% |
| With EDM aux data | 30.80% | 25.86% |
| Paper (baseline) | 57.34% | 34.25% |
| Paper (with aux) | 61.04% | 36.98% |

## Setup
```bash
pip install -r requirements.txt
```

## Train
```bash
python train.py
```

## Inference
```bash
python inference.py
```

## Kaggle Notebook
https://www.kaggle.com/code/irumisrar/dl-reproduce-results
https://www.kaggle.com/code/irumisrar/dl-proposed-3

