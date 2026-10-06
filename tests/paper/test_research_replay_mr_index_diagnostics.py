from __future__ import annotations

import pandas as pd

from src.data_loader import load_databento_mnq
from src.feature_engine import add_return_features, add_volatility_features
from src.paper.research_replay import (
    HMM_FEATURES,
    add_directional_features,
    add_session_information,
)
from src.research.mean_reversion.features import feature_engine as mr_feature_engine
from src.research.mean_reversion.features import volatility as mr_volatility
from src.strategies.s2r.signal import BASE_FEATURES


def _duplicate_labels(index: pd.Index) -> pd.Index:
    return index[index.duplicated(keep=False)].unique()


def _assert_index_health(name: str, frame: pd.DataFrame, expected_index: pd.Index) -> None:
    duplicate_index = _duplicate_labels(frame.index)
    duplicate_columns = _duplicate_labels(frame.columns)
    timestamp_duplicates = (
        int(frame["timestamp"].duplicated().sum())
        if "timestamp" in frame
        else int(frame["timestamp ET"].duplicated().sum())
        if "timestamp ET" in frame
        else 0
    )
    details = (
        f"stage={name}; shape={frame.shape}; index_type={type(frame.index).__name__}; "
        f"rows={len(frame)}; unique_index={frame.index.nunique()}; "
        f"duplicate_index_labels={len(duplicate_index)}; "
        f"first_duplicate_labels={duplicate_index[:20].tolist()}; "
        f"duplicate_columns={duplicate_columns[:20].tolist()}; "
        f"duplicate_timestamps={timestamp_duplicates}; "
        f"index_min={frame.index.min() if len(frame) else None}; "
        f"index_max={frame.index.max() if len(frame) else None}"
    )
    assert len(frame) == len(expected_index), details
    assert frame.index.equals(expected_index), details
    assert frame.index.is_unique, details
    assert frame.columns.is_unique, details
    assert timestamp_duplicates == 0, details


def _operand_diagnostic(name: str, operand: object) -> str:
    if isinstance(operand, (pd.Series, pd.DataFrame)):
        index = operand.index
        duplicates = _duplicate_labels(index)
        try:
            representative = operand.loc[duplicates[:20]].head(40).to_string()
        except Exception as error:  # diagnostic fallback must preserve original failure
            representative = f"could not render duplicate rows: {error!r}"
        return (
            f"operand={name}; type={type(operand).__name__}; "
            f"shape={getattr(operand, 'shape', None)}; "
            f"index_type={type(index).__name__}; index_is_unique={index.is_unique}; "
            f"duplicate_label_count={len(duplicates)}; "
            f"first_20_duplicate_labels={duplicates[:20].tolist()}; "
            f"duplicate_columns={_duplicate_labels(operand.columns).tolist() if isinstance(operand, pd.DataFrame) else []}; "
            f"representative_rows={representative if len(duplicates) else operand.head(5).to_string()}; "
            f"index_min={index.min() if len(index) else None}; "
            f"index_max={index.max() if len(index) else None}"
        )
    return f"operand={name}; type={type(operand).__name__}; value={operand!r}"


def test_bounded_replay_mr_preparation_reports_ratio_operand_labels(monkeypatch):
    """Exercise the full raw-data preparation path and diagnose ratio alignment."""
    raw = load_databento_mnq()
    market = raw.copy()
    market["timestamp ET"] = pd.to_datetime(market["timestamp ET"], errors="raise")
    market = market.sort_values("timestamp ET", kind="mergesort").reset_index(drop=True)
    market["timestamp"] = market["timestamp ET"].dt.tz_convert("UTC")
    expected_index = market.index.copy()
    _assert_index_health("canonical_sort", market, expected_index)

    for name, transform in (
        ("session_construction", add_session_information),
        ("return_features", add_return_features),
        ("volatility_features", add_volatility_features),
        ("directional_features", add_directional_features),
    ):
        market = transform(market)
        _assert_index_health(name, market, expected_index)
        assert market.columns.is_unique, (
            f"{name} introduced duplicate columns: "
            f"{_duplicate_labels(market.columns).tolist()}"
        )

    feature_columns = [
        "timestamp ET",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "market_period",
        *HMM_FEATURES,
        *BASE_FEATURES,
    ]
    market = market[feature_columns].copy()
    _assert_index_health("replay_feature_selection", market, expected_index)
    assert list(market.columns).count("realized_vol_30") == 1

    rth = market.loc[market["market_period"].eq("RTH")].copy()
    rth_index = rth.index.copy()
    _assert_index_health("rth_filter", rth, rth_index)
    assert list(rth.columns).count("realized_vol_30") == 1

    original_mr_returns = mr_feature_engine.add_return_features
    original_mr_volatility = mr_feature_engine.add_volatility_features
    original_ratio = mr_volatility.add_volatility_ratios

    def checked_mr_stage(name, original):
        def run(frame):
            before_index = frame.index.copy()
            result = original(frame)
            _assert_index_health(name, result, before_index)
            return result

        return run

    def checked_volatility_ratios(frame):
        operands = {
            "realized_vol_5": frame["realized_vol_5"],
            "realized_vol_30": frame["realized_vol_30"],
        }
        diagnostics = [_operand_diagnostic(name, value) for name, value in operands.items()]
        left = operands["realized_vol_5"]
        right = operands["realized_vol_30"]
        problems = []
        duplicate_frame_columns = _duplicate_labels(frame.columns)
        if len(duplicate_frame_columns):
            problems.append(
                "ratio input has duplicate column labels: "
                f"{duplicate_frame_columns[:20].tolist()}"
            )
        for name, operand in operands.items():
            if not isinstance(operand, pd.Series):
                problems.append(f"{name} did not select one Series")
            if isinstance(operand, (pd.Series, pd.DataFrame)) and not operand.index.is_unique:
                problems.append(f"{name} has duplicate index labels")
        if isinstance(left, (pd.Series, pd.DataFrame)) and isinstance(
            right, (pd.Series, pd.DataFrame)
        ):
            if type(left.index) is not type(right.index):
                problems.append("operand index types differ")
            if not left.index.equals(right.index):
                problems.append("operand indices differ or are not aligned")
        if problems:
            diagnostic = (
                "volatility-ratio alignment diagnostic: "
                + "; ".join(problems)
                + "\n"
                + f"ratio_input_shape={frame.shape}; "
                + f"ratio_input_index_type={type(frame.index).__name__}; "
                + f"ratio_input_index_is_unique={frame.index.is_unique}; "
                + f"ratio_input_duplicate_columns={duplicate_frame_columns[:20].tolist()}\n"
                + "\n".join(diagnostics)
            )
            raise AssertionError(diagnostic)
        return original_ratio(frame)

    monkeypatch.setattr(
        mr_feature_engine,
        "add_return_features",
        checked_mr_stage("research_mr_returns", original_mr_returns),
    )
    monkeypatch.setattr(
        mr_feature_engine,
        "add_volatility_features",
        checked_mr_stage("research_mr_volatility", original_mr_volatility),
    )
    monkeypatch.setattr(mr_volatility, "add_volatility_ratios", checked_volatility_ratios)

    mr_features = mr_feature_engine.build_mean_reversion_features(rth.copy())
    _assert_index_health("research_mr_complete", mr_features, rth_index)
