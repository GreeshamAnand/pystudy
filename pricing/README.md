# Pricing Data Science — Causal Pricing in Two-Sided Marketplaces

A hands-on, notebook-driven study track that builds from **predictive** pricing toward
**causal** pricing in **two-sided marketplaces** (think Uber and Airbnb).

The arc, in one sentence: *predicting a price from features is easy and useful, but it does
not tell you how demand will **respond** when you **change** the price — and in a
supply-and-demand world, naively reading that response off historical data is badly biased.
Causal inference is how you fix it.*

## Why this matters

In a marketplace, price is not an exogenous knob — it is set by, and reacts to, the same
supply-and-demand forces that drive the quantity you are trying to predict. A model that
fits *observed* prices learns equilibrium, not causation. Setting prices well requires the
**causal** price → demand relationship (the elasticity), which usually requires
**experiments** or **instrumental variables**, not a better predictor.

## Learning path

| # | Notebook | What you learn |
|---|----------|----------------|
| 01 | [`01_predictive_pricing.ipynb`](01_predictive_pricing.ipynb) | Predict price & booking from features (Airbnb's playbook). Build a GBM, see its ceiling. |
| 02 | [`02_elasticity_supply_demand.ipynb`](02_elasticity_supply_demand.ipynb) | Demand curves, **price elasticity**, elastic vs. inelastic regimes, revenue-max price, market equilibrium. |
| 03 | [`03_causal_problem_endogeneity.ipynb`](03_causal_problem_endogeneity.ipynb) | Why naive elasticity is **biased** (endogeneity/simultaneity). Fix with **IV/2SLS** and **DoWhy**. |
| 04 | [`04_causal_marketplace_pricing.ipynb`](04_causal_marketplace_pricing.ipynb) | Two-sided marketplaces, **interference / SUTVA**, heterogeneous effects with **EconML**, toward a causal pricing structure. |

Every notebook **runs end-to-end on synthetic data with no downloads**. The synthetic
generators live in [`utils/datagen.py`](utils/datagen.py) and return *known* ground-truth
parameters so you can check what a model recovers against the truth. Sections that use real
**Kaggle** data are clearly marked and optional (they need network + credentials).

## How to run

```bash
pip install -r requirements.txt          # or the root requirements.txt
jupyter lab                              # then open pricing/0X_*.ipynb in order

# Headless sanity check (synthetic paths only):
jupyter nbconvert --to notebook --execute pricing/01_predictive_pricing.ipynb
```

### Real data (optional)

Notebook 01 can pull the **NYC Airbnb Open Data** set from Kaggle via
[`kagglehub`](https://github.com/Kaggle/kagglehub). You need Kaggle API credentials
(`~/.kaggle/kaggle.json`). The download cell is wrapped in `try/except`, so the notebook
still completes without network — it just skips the real-data comparison.

## Key references (blogs & research)

**Airbnb — predictive / dynamic pricing**
- Ye et al., *Customized Regression Model for Airbnb Dynamic Pricing*, **KDD 2018** —
  [paper page](https://www.kdd.org/kdd2018/accepted-papers/view/customized-regression-model-for-airbnb-dynamic-pricing) ·
  [ACM DL](https://dl.acm.org/doi/10.1145/3219819.3219830) ·
  [the morning paper summary](https://blog.acolyer.org/2018/10/03/customized-regression-model-for-airbnb-dynamic-pricing/)
- Airbnb Engineering, *Learning Market Dynamics for Optimal Pricing* —
  [Medium](https://medium.com/airbnb-engineering/learning-market-dynamics-for-optimal-pricing-97cffbcc53e3)

**Uber — marketplace, surge & causal ML**
- *Practical Marketplace Optimization at Uber Using Causally-Informed Machine Learning*,
  **arXiv:2407.19078** — [paper](https://arxiv.org/html/2407.19078v1)
- Uber Engineering, *Engineering Uber's Next-Gen Surge Pricing* —
  [blog](https://www.uber.com/blog/research/the-economics-of-surge-pricing/)
- Cohen et al., *Using Big Data to Estimate Consumer Surplus: The Case of Uber*,
  **NBER w22627** — [PDF](https://www.nber.org/system/files/working_papers/w22627/w22627.pdf)
- Castillo, *Who Benefits from Surge Pricing?* —
  [PDF](https://economics.sas.upenn.edu/system/files/2020-01/JMP_Castillo.pdf)

**Causal inference toolkit**
- [DoWhy](https://www.pywhy.org/dowhy/) — model → identify → estimate → **refute**
- [EconML](https://econml.azurewebsites.net/) — heterogeneous treatment effects (DML, causal forests)
- [linearmodels `IV2SLS`](https://bashtage.github.io/linearmodels/iv/index.html) — instrumental variables
- Cunningham, *Causal Inference: The Mixtape* — [free online](https://mixtape.scunning.com/)
- Facure, *Causal Inference for the Brave and True* — [free online](https://matheusfacure.github.io/python-causality-handbook/)

**Datasets (Kaggle)**
- [NYC Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) (`dgomonov/new-york-city-airbnb-open-data`)
