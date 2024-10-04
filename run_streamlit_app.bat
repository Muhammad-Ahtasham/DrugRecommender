@echo off
:: Activate the Conda environment(drenv)
echo Opening Conda environment: drenv
call conda activate drenv

:: Running The App
echo Runnign the App.py via streamlit
call streamlit run App.py

:: Notify completion
echo Everthing is completed and saved
call pause
