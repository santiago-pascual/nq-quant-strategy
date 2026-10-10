"""Reuse the actual ORB/HMM crash fixture with isolated hardened checkpoint IO."""
from pathlib import Path
import runpy
import subprocess


def test_hardened_store_real_orb_hmm_crash_and_catchup(monkeypatch):
    from src.paper.realtime_checkpoint import AtomicCheckpointStore
    from src.paper.retry_checkpoint import RetrySafeCheckpointStore
    monkeypatch.setattr(AtomicCheckpointStore, 'save', RetrySafeCheckpointStore.save)
    original_run = subprocess.run
    def isolated_child(args, *positional, **kwargs):
        if isinstance(args,list) and len(args)>2 and args[1]=='-c' and 'orb-hard-crash-test' in args[2]:
            args=list(args)
            args[2] = ('from src.paper.realtime_checkpoint import AtomicCheckpointStore\n'
                       'from src.paper.retry_checkpoint import RetrySafeCheckpointStore\n'
                       'AtomicCheckpointStore.save = RetrySafeCheckpointStore.save\n')+args[2]
        return original_run(args,*positional,**kwargs)
    monkeypatch.setattr(subprocess,'run',isolated_child)
    fixture=runpy.run_path(str(Path(__file__).with_name('test_ibkr_paper_recovery.py')))
    # Strategies, fitted HMM/refits, order/fill assertions and economic equality
    # remain exactly those in the original fixture. Only isolated IO is changed.
    fixture['test_hard_crash_after_real_orb_fill_replays_bar_without_duplicate_execution']()
