uv run ./experiments/exp_02_diffrax_tossobjects/run.py xml.system="ball"

:: Billiard experiment
uv run experiments/exp_03_billiard/run.py

:: Billiard ecperiment with backsolve adjoint
uv run experiments/exp_03_billiard/run.py --config-name=config_backsolve
