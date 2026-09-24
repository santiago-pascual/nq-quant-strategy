Risk Engine checkpoint
=======================

Files:
- types.py
- engine.py
- test_engine.py

Copy:
    src/risk/types.py
    src/risk/engine.py
    tests/risk/test_engine.py

Run:
    pytest -q tests/risk

This first block is deliberately independent from the execution engine.
It implements deterministic sizing and hard risk limits before broker integration.
