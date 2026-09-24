Risk <-> Execution Integration
===============================

Copy:
    src/portfolio/__init__.py
    src/portfolio/risk_execution.py
    tests/portfolio/test_risk_execution.py

Run:
    pytest -q tests/portfolio

Architecture:
    Strategy
       |
       v
    RiskExecutionCoordinator
       |       | RiskEngine
       |
       v
    ExecutionEngine

The risk policy values in these tests are TEST VALUES ONLY.
They are NOT the frozen Topstep 50K policy.

Do not replace them with Topstep values yet.
When the project reaches the actual Risk Policy stage, stop and
review the current Topstep rules/account constraints before freezing
production parameters.
