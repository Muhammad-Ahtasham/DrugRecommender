@echo off
:: Create a new conda environment named drenv
echo Creating a new Conda environment: drenv
call conda create -n drenv python=3.10 pip -y

:: Activate the new environment
echo Activating the drenv environment
call conda activate drenv

:: Install the requirements using pip (assumes requirements.txt exists in the same directory)
echo Installing requirements via pip
call pip install -r requirements.txt

:: Notify completion
echo Environment setup complete!
call pause
