@echo off
rem Records open interest, funding and order flow for 20 Phemex perps into .\data, restarting after any crash.
cd /d "%~dp0"
:loop
python -m scalper record-all
echo recorder stopped; restarting in 10 s (close this window to quit)
timeout /t 10 /nobreak >nul
goto loop
