# EDA Mini Project B — Time-Series Forecasting Starter

Student: Younis  
Student ID: PG12S2540600

This repository contains a starter Streamlit app for Mini Project B using a cleaned time-series dataset sample.

## Files

- `app.py` — one-file Streamlit app
- `requirements.txt` — Python package requirements
- `data/dataset_sample.csv` — cleaned dataset sample used by the app
- `README.md` — setup and submission instructions

## Run locally

1. Install Python 3.12 or later.
2. Open a terminal in this project folder.
3. Install requirements:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```

## OpenRouter API key

The app does not hardcode any API key. For the AI grader, provide your key using one of these options:

1. Streamlit Secrets: `OPENROUTER_API_KEY`
2. Environment variable: `OPENROUTER_API_KEY`
3. Password input field inside the app

## Deploy on Streamlit Community Cloud

1. Create a public GitHub repository named `EDA-ProjectB-PG12S2540600`.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - the full `data` folder containing `dataset_sample.csv`
3. Go to Streamlit Community Cloud.
4. Create a new app.
5. Connect the GitHub repository.
6. Select branch `main`.
7. Set main file path to `app.py`.
8. Deploy.

## What to submit

Submit these items to your instructor:

- Streamlit deployed app URL
- GitHub repository URL
- Exported `submission.json` from the app
- Exported `project_card.md` from the app
- Screenshots requested by the project brief
