# Kaggle Playground S6E9 — Predicting Electric Vehicle Purchases

Solution walkthrough for [Playground Series S6E9 · Predicting Electric Vehicle Purchases](https://www.kaggle.com/competitions/playground-series-s6e9) (binary classification, ROC AUC).

| | Public | Private |
|---|---:|---:|
| Final submission | 0.94683 (30th) | **0.94567 (40th / 3,575 teams)** |
| Best private among our submissions | 0.94680 | 0.94569 |

## Notebook: [`s6e9_ev_purchase_solution.ipynb`](s6e9_ev_purchase_solution.ipynb)

The notebook is written in Korean and ships with all outputs and figures.

1. **Exploratory data analysis**
   - Original (10k rows) vs synthetic data distributions
   - Reverse-engineering the data generator: income floor at $30,000 and commute floor at 5 km; the synthetic data amplifies these spikes (income $30,000: 5.7% → 9.2%, commute 5 km: 7.9% → 21.6%)
   - The target recipe alone reaches AUC 0.938 on train
   - Income values tokenized with the GPT-2 tokenizer: the leading token shifts the purchase rate away from the recipe in a systematic way
   - Adversarial validation (train vs test AUC 0.499): same distribution, so CV can be trusted
2. **Models** (all on the same 5-fold OOF split)

   | Model | OOF AUC |
   |---|---:|
   | A. XGBoost on raw features | 0.94190 |
   | B. XGBoost with the recipe as base margin | 0.94194 |
   | C. Generator-aware ridge logistic regression (simplified) | 0.94614 |
   | **D. XGBoost boosted from C's logit** | **0.94644** |
   | E. Level-2 logistic regression stacker (A–D) | 0.94642 |

3. **Competition results**: public vs private score for each submission, and what actually held up on the private leaderboard
4. **Lessons learned**

## Key takeaways

- Playground data comes from a generator trained on a small original dataset. Reverse-engineering it (the target recipe, the token structure of numbers) gives a strong starting point.
- Feeding a strong simple model's logit to a GBDT as its **base margin** lets the trees focus on corrections.
- A **level-2 stacker** beats hand-picked fixed blend weights, as long as its inputs are fully nested out-of-fold predictions and are rank-transformed within each fold.
- The private leaderboard followed CV, not tiny public differences. Pick final submissions by OOF score.

## Running

Competition data is not included, per the competition rules. Place these files under `data/`:

- [Competition data](https://www.kaggle.com/competitions/playground-series-s6e9/data): `train.csv`, `test.csv`
- [Original dataset](https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety): `EV_Adoption_and_Range_Anxiety_Dataset.csv`

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute s6e9_ev_purchase_solution.ipynb
```

Runs in about 6 minutes on CPU (Apple M4, 10 cores). On Kaggle, files under `/kaggle/input` are found automatically.

## Acknowledgements

- Chris Deotte — [EDA: Original Data Insights](https://www.kaggle.com/code/cdeotte/fable-5-1-eda-original-data-insights), [XGB Starter](https://www.kaggle.com/code/cdeotte/fable-5-1-xgb-starter), [GPU Logistic Regression Stacker (S6E6)](https://www.kaggle.com/code/cdeotte/gpu-logistic-regression-stacker)
- heuljax — [Generator-Aware Ridge Logistic Regression](https://www.kaggle.com/code/heuljax/kps6e09-generator-aware-ridge-logistic-regression)
- goodpjw2008 — [LR-Margin GBDT + OOF Stack](https://www.kaggle.com/code/goodpjw2008/s6e9-lr-margin-gbdt-oof-stack-lb-0-94675)
