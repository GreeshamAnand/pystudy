"""
Synthetic data generators for the pricing learning series.

Every generator returns a tuple of ``(DataFrame, ground_truth)`` where
``ground_truth`` is a dict of the *true* parameters used to build the data.
Because we know the true elasticity / treatment effect, we can compare what a
naive model recovers against reality -- which is the whole point of the causal
notebooks (03 and 04).

Style mirrors the existing teaching code in ``src/MC.py``: small, readable,
documented functions with an example ``__main__`` block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Notebook 01 -- Airbnb-style listings for predictive pricing
# ---------------------------------------------------------------------------
def simulate_listings(n: int = 5000, seed: int = 0):
    """Generate Airbnb-style listings with features, a posted price and a
    booking outcome.

    The data-generating process is intentionally simple and *known*:

        latent_value = f(bedrooms, location, reviews, season, ...)
        price        = latent_value scaled to dollars + noise
        booked       = Bernoulli(sigmoid(value - price sensitivity))

    Returns
    -------
    (df, ground_truth)
        ``df`` has feature columns plus ``price`` and ``booked``.
        ``ground_truth`` records the coefficients of the latent value model.
    """
    rng = np.random.default_rng(seed)

    bedrooms = rng.integers(1, 5, size=n)
    bathrooms = np.clip(bedrooms + rng.integers(-1, 2, size=n), 1, None)
    accommodates = bedrooms * 2 + rng.integers(0, 3, size=n)
    location_score = rng.uniform(0, 1, size=n)          # 0 = remote, 1 = prime
    review_score = rng.uniform(3.0, 5.0, size=n)
    num_reviews = rng.poisson(20, size=n)
    is_entire_home = rng.binomial(1, 0.6, size=n)
    # Seasonality: day-of-year -> a smooth demand multiplier (summer peak).
    day_of_year = rng.integers(0, 365, size=n)
    season = 1.0 + 0.3 * np.sin((day_of_year / 365.0) * 2 * np.pi - np.pi / 2)

    # Known latent value coefficients (the "ground truth").
    coef = {
        "intercept": 40.0,
        "bedrooms": 22.0,
        "bathrooms": 12.0,
        "accommodates": 6.0,
        "location_score": 90.0,
        "review_score": 10.0,
        "num_reviews": 0.15,
        "is_entire_home": 25.0,
    }

    latent_value = (
        coef["intercept"]
        + coef["bedrooms"] * bedrooms
        + coef["bathrooms"] * bathrooms
        + coef["accommodates"] * accommodates
        + coef["location_score"] * location_score
        + coef["review_score"] * review_score
        + coef["num_reviews"] * num_reviews
        + coef["is_entire_home"] * is_entire_home
    )
    latent_value *= season

    # Hosts post a price near the latent value, with idiosyncratic noise.
    price = latent_value * rng.normal(1.0, 0.15, size=n)
    price = np.clip(price, 20, None).round(0)

    # Booking probability falls when price exceeds the listing's latent value.
    value_gap = (latent_value - price) / 50.0
    booking_prob = 1.0 / (1.0 + np.exp(-value_gap))
    booked = rng.binomial(1, booking_prob)

    df = pd.DataFrame(
        {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "accommodates": accommodates,
            "location_score": location_score.round(3),
            "review_score": review_score.round(2),
            "num_reviews": num_reviews,
            "is_entire_home": is_entire_home,
            "day_of_year": day_of_year,
            "season_multiplier": season.round(3),
            "price": price,
            "booked": booked,
        }
    )
    ground_truth = {"coef": coef, "booking_prob": booking_prob}
    return df, ground_truth


# ---------------------------------------------------------------------------
# Notebook 02 -- Demand curves with a KNOWN elasticity
# ---------------------------------------------------------------------------
# A constant-elasticity demand curve:  Q = A * P ** elasticity
# Elasticity is negative; |elasticity| > 1 -> elastic, < 1 -> inelastic.
ELASTICITY_PRESETS = {
    "very_inelastic": -0.3,   # e.g. insulin, gasoline (short run)
    "inelastic": -0.7,
    "unit": -1.0,
    "elastic": -1.8,          # e.g. a specific brand with many substitutes
    "very_elastic": -3.0,     # e.g. one seller's identical commodity
}


def demand_at_price(price, elasticity: float, scale: float = 1e5):
    """Constant-elasticity demand:  Q = scale * P**elasticity."""
    price = np.asarray(price, dtype=float)
    return scale * price ** elasticity


def simulate_demand_curve(
    elasticity: float = -1.8,
    p_min: float = 10.0,
    p_max: float = 200.0,
    n_points: int = 60,
    scale: float = 1e5,
    noise: float = 0.0,
    seed: int = 0,
):
    """Tabulate a demand curve and the revenue at each price point.

    Returns
    -------
    (df, ground_truth)
        ``df`` columns: ``price``, ``quantity``, ``revenue``.
        ``ground_truth``: elasticity, scale, and the revenue-maximising price.
    """
    rng = np.random.default_rng(seed)
    price = np.linspace(p_min, p_max, n_points)
    quantity = demand_at_price(price, elasticity, scale)
    if noise > 0:
        quantity = quantity * rng.normal(1.0, noise, size=price.shape)
    revenue = price * quantity

    df = pd.DataFrame({"price": price, "quantity": quantity, "revenue": revenue})

    # For constant elasticity, revenue is monotone in price (no interior max)
    # unless |elasticity| == 1. We still report the argmax over the grid so the
    # notebook can discuss the elastic/inelastic intuition concretely.
    rev_max_price = float(df.loc[df["revenue"].idxmax(), "price"])
    ground_truth = {
        "elasticity": elasticity,
        "scale": scale,
        "revenue_max_price_on_grid": rev_max_price,
    }
    return df, ground_truth


def simulate_linear_market(
    seed: int = 0,
    n_points: int = 60,
    demand_intercept: float = 200.0,
    demand_slope: float = -1.5,
    supply_intercept: float = 20.0,
    supply_slope: float = 1.0,
):
    """A textbook linear supply & demand market with a clean equilibrium.

    Demand:  Qd = demand_intercept + demand_slope * P
    Supply:  Qs = supply_intercept + supply_slope * P
    Equilibrium where Qd == Qs.

    Returns ``(df, ground_truth)`` with the equilibrium price/quantity.
    """
    price = np.linspace(1, 150, n_points)
    qd = demand_intercept + demand_slope * price
    qs = supply_intercept + supply_slope * price
    df = pd.DataFrame({"price": price, "quantity_demanded": qd, "quantity_supplied": qs})

    eq_price = (demand_intercept - supply_intercept) / (supply_slope - demand_slope)
    eq_qty = demand_intercept + demand_slope * eq_price
    ground_truth = {
        "equilibrium_price": float(eq_price),
        "equilibrium_quantity": float(eq_qty),
        "demand": (demand_intercept, demand_slope),
        "supply": (supply_intercept, supply_slope),
    }
    return df, ground_truth


# ---------------------------------------------------------------------------
# Notebook 03 -- Endogenous prices (the core causal problem)
# ---------------------------------------------------------------------------
def simulate_endogenous_prices(
    n: int = 4000,
    true_elasticity: float = -1.5,
    confounder_strength: float = 1.2,
    instrument_strength: float = 0.8,
    seed: int = 0,
):
    """Generate price/quantity data where an UNOBSERVED demand shock drives
    both, so a naive regression of log-quantity on log-price is biased.

    Structural (log) system:

        demand_shock ~ N(0, 1)                       # unobserved confounder
        cost_shifter ~ N(0, 1)                       # the INSTRUMENT (supply side)

        log_price    = a + instrument_strength*cost_shifter
                         + confounder_strength*demand_shock + noise_p
        log_quantity = b + true_elasticity*log_price
                         + confounder_strength*demand_shock + noise_q

    Because ``demand_shock`` enters both equations, OLS of log_quantity on
    log_price is biased upward (toward 0 / positive). ``cost_shifter`` affects
    quantity ONLY through price -> a valid instrument for 2SLS.

    Returns
    -------
    (df, ground_truth)
        ``df`` columns: ``log_price``, ``log_quantity``, ``price``,
        ``quantity``, ``cost_shifter`` (observed instrument). The
        ``demand_shock`` is intentionally returned too so notebooks can *show*
        the confounding, but it must NOT be used as a regressor in the naive fit.
    """
    rng = np.random.default_rng(seed)

    demand_shock = rng.normal(0, 1, size=n)            # unobserved in practice
    cost_shifter = rng.normal(0, 1, size=n)            # observed instrument

    log_price = (
        3.5
        + instrument_strength * cost_shifter
        + confounder_strength * demand_shock
        + rng.normal(0, 0.3, size=n)
    )
    log_quantity = (
        8.0
        + true_elasticity * log_price
        + confounder_strength * demand_shock
        + rng.normal(0, 0.3, size=n)
    )

    df = pd.DataFrame(
        {
            "log_price": log_price,
            "log_quantity": log_quantity,
            "price": np.exp(log_price),
            "quantity": np.exp(log_quantity),
            "cost_shifter": cost_shifter,
            "demand_shock": demand_shock,  # "unobserved" -- for illustration only
        }
    )
    ground_truth = {
        "true_elasticity": true_elasticity,
        "confounder_strength": confounder_strength,
        "instrument_strength": instrument_strength,
    }
    return df, ground_truth


# ---------------------------------------------------------------------------
# Notebook 04 -- Two-sided marketplace with heterogeneous price sensitivity
# ---------------------------------------------------------------------------
def simulate_two_sided(
    n: int = 6000,
    base_elasticity: float = -1.2,
    seed: int = 0,
):
    """Simulate a ride-hailing-style two-sided marketplace session log with a
    KNOWN, heterogeneous treatment effect of price (surge multiplier).

    Each row is a market-session for a rider segment. The surge multiplier is
    the "treatment". The causal effect of surge on the rider's book probability
    varies by segment (commuters are less price sensitive than leisure riders),
    which is exactly the heterogeneity EconML should recover.

    A confounder (``demand_intensity``) raises both the surge price AND booking
    propensity, so a naive correlation understates how much surge suppresses
    demand -- the observational trap, again.

    Returns
    -------
    (df, ground_truth)
        ``df`` includes covariates X, the treatment ``surge``, the outcome
        ``booked``, and per-row ``true_cate`` (true marginal effect of surge).
    """
    rng = np.random.default_rng(seed)

    # Rider segment: 0 = leisure (price sensitive), 1 = commuter (less so).
    is_commuter = rng.binomial(1, 0.45, size=n)
    rider_income = rng.normal(0, 1, size=n)            # standardized
    trip_distance = rng.gamma(2.0, 2.0, size=n)
    time_of_day = rng.uniform(0, 24, size=n)
    is_peak = ((time_of_day > 7) & (time_of_day < 10)) | (
        (time_of_day > 16) & (time_of_day < 19)
    )

    # Confounder: latent demand intensity (e.g. a concert lets out). Unobserved
    # drivers raise surge; here we expose a noisy proxy but keep true intensity.
    demand_intensity = rng.normal(0, 1, size=n) + 0.6 * is_peak

    # Surge multiplier (the treatment) rises with demand intensity + noise.
    surge = 1.0 + 0.5 * np.maximum(demand_intensity, 0) + rng.normal(0, 0.15, size=n)
    surge = np.clip(surge, 1.0, 3.5)

    # Heterogeneous true effect of surge on booking (more negative = more
    # sensitive). Commuters & higher income are LESS price sensitive.
    true_cate = (
        base_elasticity
        + 0.7 * is_commuter
        + 0.25 * rider_income
        - 0.10 * trip_distance / trip_distance.std()
    )
    # Keep effects negative (surge suppresses demand) but heterogeneous.
    true_cate = np.minimum(true_cate, -0.1)

    # Latent booking utility: confounder raises booking; surge suppresses it
    # via the (heterogeneous) causal effect.
    utility = (
        0.4
        + 0.8 * demand_intensity            # confounding path
        + 0.3 * is_commuter
        + 0.2 * rider_income
        + true_cate * (surge - 1.0)         # causal effect of surge
        + rng.normal(0, 0.5, size=n)
    )
    booked = (utility > 0).astype(int)

    df = pd.DataFrame(
        {
            "is_commuter": is_commuter,
            "rider_income": rider_income.round(3),
            "trip_distance": trip_distance.round(2),
            "time_of_day": time_of_day.round(2),
            "is_peak": is_peak.astype(int),
            "demand_intensity": demand_intensity.round(3),  # confounder proxy
            "surge": surge.round(3),
            "booked": booked,
            "true_cate": true_cate.round(4),
        }
    )
    ground_truth = {
        "base_elasticity": base_elasticity,
        "ate": float(np.mean(true_cate)),
        "feature_cols": [
            "is_commuter",
            "rider_income",
            "trip_distance",
            "time_of_day",
            "is_peak",
        ],
        "confounder_col": "demand_intensity",
        "treatment_col": "surge",
        "outcome_col": "booked",
    }
    return df, ground_truth


if __name__ == "__main__":
    # Quick smoke test of every generator.
    for name, fn in [
        ("listings", lambda: simulate_listings(1000)),
        ("demand_curve", lambda: simulate_demand_curve(-1.8)),
        ("linear_market", lambda: simulate_linear_market()),
        ("endogenous", lambda: simulate_endogenous_prices(2000)),
        ("two_sided", lambda: simulate_two_sided(2000)),
    ]:
        df, gt = fn()
        print(f"\n=== {name} ===")
        print(df.head(3).to_string())
        print("ground_truth keys:", list(gt.keys()))
