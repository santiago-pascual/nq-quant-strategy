from pathlib import Path
import pandas as pd

from src.paper.autonomous_runner import (
    AutonomousRunConfig,
    AutonomousPaperRunner,
    load_canonical_raw_mnq,
    load_frozen_research_hmm_windows,
)
from src.paper.context_adapter import PaperMarketContextAdapter
from src.paper.engine import PaperEngineConfig, PaperTradingEngine
from src.paper.logger import PaperEventLogger
from src.broker import InMemoryBrokerAdapter
from src.execution import ExecutionEngine
from src.portfolio.conflict import PortfolioConflictEngine
from src.risk import RiskEngine, RiskLimits

from src.strategies.mean_reversion.config import MRL1_CONFIG, MRS2_CONFIG
from src.strategies.mean_reversion.strategy import MeanReversionStrategy
from src.strategies.orb.strategy import ORBStrategy
from src.strategies.s2r.strategy import S2RStrategy


ROOT = Path.cwd()

print("Loading canonical MNQ...")
df = load_canonical_raw_mnq()

print(f"Rows: {len(df):,}")
print(f"First: {df['timestamp'].min()}")
print(f"Last:  {df['timestamp'].max()}")

# Use the frozen 22-window manifest.
schedule = ROOT / "src" / "paper" / "config" / "research_07_hmm_window_schedule.csv"
windows = load_frozen_research_hmm_windows(str(schedule))

print(f"Frozen HMM windows: {len(windows)}")

# Pick one real RTH session late enough to have substantial causal history.
# We intentionally do NOT replay the whole dataset yet.
session_dates = (
    pd.to_datetime(df["timestamp"], utc=True)
    .dt.tz_convert("America/New_York")
    .dt.date
)

# Pick the first session after 2020-06-23 that has a full RTH-sized sample.
tmp = df.copy()
tmp["_ny_date"] = session_dates
tmp["_ny_time"] = pd.to_datetime(tmp["timestamp"], utc=True).dt.tz_convert(
    "America/New_York"
).dt.time

candidate_dates = sorted(tmp.loc[tmp["_ny_date"] >= pd.Timestamp("2020-06-23").date(), "_ny_date"].unique())

target_date = candidate_dates[100]

print(f"Smoke-test session: {target_date}")

# Keep all history through the target session.
replay_df = tmp[tmp["_ny_date"] <= target_date].drop(
    columns=["_ny_date", "_ny_time"]
).copy()

print(f"Replay rows through target session: {len(replay_df):,}")

context_adapter = PaperMarketContextAdapter()

context_adapter.configure_research_windows(windows)

strategies = [
    MeanReversionStrategy(MRL1_CONFIG),
    MeanReversionStrategy(MRS2_CONFIG),
    S2RStrategy(),
    ORBStrategy(),
]

broker = InMemoryBrokerAdapter()

engine = PaperTradingEngine(
    strategies=strategies,
    execution=ExecutionEngine(),
    risk=RiskEngine(
        RiskLimits(
            risk_per_trade=250.0,
            max_total_risk=10_000.0,
            max_daily_loss=5_000.0,
            max_concurrent_positions=4,
            max_daily_trades=10,
            max_contracts=1,
        )
    ),
    conflict=PortfolioConflictEngine(max_concurrent_positions=4),
    broker=broker,
    logger=PaperEventLogger(
        ROOT / "results" / "paper" / "smoke_test" / "events.jsonl"
    ),
    context_adapter=context_adapter,
    config=PaperEngineConfig(
        initial_equity=50_000.0,
        point_value=2.0,
        commission_per_contract=1.0,
        automatic_simulated_fills=True,
    ),
)

engine.connect()

runner = AutonomousPaperRunner(
    paper_engine=engine,
    context_adapter=context_adapter,
    config=AutonomousRunConfig(
        output_dir=ROOT / "results" / "paper" / "smoke_test",
        progress_every_bars=100_000,
    ),
)

stats = runner.run(replay_df)

print()
print("=" * 80)
print("SMOKE TEST RESULT")
print("=" * 80)
print(f"Processed bars: {stats.processed_bars:,}")
print(f"Fills:          {getattr(stats, 'fills', 'n/a')}")
print(f"Account equity: {engine.account_equity}")
print(f"Realized P&L:   {engine.realized_pnl}")
print(f"Commissions:    {engine.commissions}")
print(f"Closed pos.:    {len(engine.execution.get_closed_positions())}")
print(f"Open pos.:      {len(engine.execution.get_positions())}")
print("=" * 80)
