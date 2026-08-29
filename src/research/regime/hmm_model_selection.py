from __future__ import annotations

import math

import pandas as pd

from src.data_loader import load_data
from src.models.regime import (
    HMM_FEATURES,
    VolatilityRegimeModel,
)


def calculate_information_criteria(
    model,
    data,
):
    """
    Calculate AIC and BIC for a fitted Gaussian HMM.
    """

    log_likelihood = model.score(data)

    n_states = model.n_components
    n_features = model.n_features

    # Initial state probabilities
    initial_state_parameters = n_states - 1

    # Transition matrix
    transition_parameters = n_states * (n_states - 1)

    # Mean vector for each state
    mean_parameters = n_states * n_features

    # Full covariance matrix for each state
    covariance_parameters = n_states * n_features * (n_features + 1) / 2

    n_parameters = (
        initial_state_parameters
        + transition_parameters
        + mean_parameters
        + covariance_parameters
    )

    n_observations = len(data)

    aic = 2 * n_parameters - 2 * log_likelihood

    bic = n_parameters * math.log(n_observations) - 2 * log_likelihood

    return aic, bic


def evaluate_model(
    df: pd.DataFrame,
    n_states: int,
) -> dict:

    model = VolatilityRegimeModel(
        n_states=n_states,
        random_state=42,
    )

    model.fit(df)

    data = model.prepare_data(df)

    standardized_data = model.standardize(
        data,
        fit=False,
    )

    states = model.predict_states(df)

    rth = df.loc[
        states.index,
        HMM_FEATURES
        + [
            "future_vol_5",
            "future_vol_15",
            "future_vol_30",
        ],
    ].copy()

    rth["hmm_state"] = states

    state_counts = states.value_counts().sort_index()

    state_proportions = states.value_counts(normalize=True).sort_index()

    future_volatility = rth.groupby("hmm_state")[
        [
            "future_vol_5",
            "future_vol_15",
            "future_vol_30",
        ]
    ].mean()

    aic, bic = calculate_information_criteria(
        model=model.model,
        data=standardized_data,
    )

    log_likelihood = model.model.score(standardized_data)

    return {
        "model": model,
        "states": states,
        "state_counts": state_counts,
        "state_proportions": state_proportions,
        "future_volatility": future_volatility,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
    }


def evaluate_model_stability(
    df: pd.DataFrame,
    n_states: int,
    seeds: list[int],
) -> list[dict]:

    results = []

    for seed in seeds:
        model = VolatilityRegimeModel(
            n_states=n_states,
            random_state=seed,
        )

        model.fit(df)

        data = model.prepare_data(df)

        standardized_data = model.standardize(
            data,
            fit=False,
        )

        states = model.predict_states(df)

        state_proportions = states.value_counts(normalize=True).sort_index().tolist()

        results.append(
            {
                "seed": seed,
                "converged": (model.model.monitor_.converged),
                "iterations": (model.model.monitor_.iter),
                "log_likelihood": (model.model.score(standardized_data)),
                "state_proportions": (state_proportions),
            }
        )

    return results


def main():

    df = load_data()

    # ---------------------------------------------------------
    # 1. Compare 2, 3, and 4-state HMMs
    # ---------------------------------------------------------

    for n_states in [2, 3, 4]:
        print(f"\n{'=' * 60}")

        print(f"HMM WITH {n_states} STATES")

        print(f"{'=' * 60}")

        result = evaluate_model(
            df=df,
            n_states=n_states,
        )

        print(
            "\nConverged:",
            result["model"].model.monitor_.converged,
        )

        print(
            "Iterations:",
            result["model"].model.monitor_.iter,
        )

        print(
            "Log likelihood:",
            result["log_likelihood"],
        )

        print(
            "AIC:",
            result["aic"],
        )

        print(
            "BIC:",
            result["bic"],
        )

        print("\nState proportions:")
        print(result["state_proportions"])

        print("\nFuture volatility by state:")

        print(result["future_volatility"])

        print("\nTransition matrix:")

        print(
            pd.DataFrame(
                result["model"].model.transmat_,
                index=[f"state_{i}" for i in range(n_states)],
                columns=[f"state_{i}" for i in range(n_states)],
            )
        )

    # ---------------------------------------------------------
    # 2. Stability across random seeds
    # ---------------------------------------------------------

    seeds = [42, 7, 21, 100]

    for n_states in [2, 3, 4]:
        print(f"\n{'#' * 60}")

        print(f"STABILITY TEST — {n_states} STATES")

        print(f"{'#' * 60}")

        stability_results = evaluate_model_stability(
            df=df,
            n_states=n_states,
            seeds=seeds,
        )

        for result in stability_results:
            print(
                "\nSeed:",
                result["seed"],
            )

            print(
                "Converged:",
                result["converged"],
            )

            print(
                "Iterations:",
                result["iterations"],
            )

            print(
                "Log likelihood:",
                result["log_likelihood"],
            )

            print(
                "State proportions:",
                result["state_proportions"],
            )


if __name__ == "__main__":
    main()
