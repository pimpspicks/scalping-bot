@echo off
rem Paper trades the two three-move setups (2h move-1 pullback, 2h move-3 fade) on 20 Phemex perps, restarting
rem after any crash. Results: paper-3move\trades.csv. No orders are ever sent.
cd /d "%~dp0"
:loop
python -m scalper paper-3move
echo paper bot stopped; restarting in 60 s (close this window to quit)
timeout /t 60 /nobreak >nul
goto loop
