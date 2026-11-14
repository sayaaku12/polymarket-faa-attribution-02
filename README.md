# Polymarket FAA Attribution Forecaster — make.fun bounty entry

Offline quantile forecaster (Python 3.13, standard library only) that predicts Polymarket's FAA outage attribution prices minute by minute, 4 hours ahead.

- `main.py` — the forecaster (deterministic, no network at runtime)

Practice (6 kit cases, NVIDIA CEO event): pinball loss 0.276669 vs house baseline 0.363589 (−23.9%, better on all 6 cases). FAA out-of-sample: 0.660 vs 0.803 (−17.8%).

Built by an AI coding agent for the make.fun bounty. MIT licensed.
