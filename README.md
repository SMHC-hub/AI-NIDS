# AI-NIDS: Explainable AI Network Intrusion Detection System

An explainable machine learning system for detecting malicious network traffic, understanding model predictions, and generating AI-assisted security analysis reports.

AI-NIDS combines **Random Forest classification**, **SHAP explainability**, **Groq-powered SOC analysis**, and an interactive **Streamlit dashboard**.

The project uses the CIC-IDS2017 dataset and focuses on making intrusion detection results easier to interpret, rather than presenting predictions as confirmed security incidents.

## Key Features

- **Network intrusion detection:** Classifies network flows as benign or malicious using a Random Forest model.

- **Explainable predictions:** Uses SHAP to identify which network features influence individual predictions.

- **AI-assisted SOC reports:** Uses Groq to generate security analysis based on model predictions and selected SHAP evidence.

- **Interactive dashboard:** Provides a Streamlit interface for exploring predictions, confidence scores, and explanations.

- **Automated testing:** Includes seven tests covering model artifacts, predictions, SHAP explanations, and Groq failure handling.

- **Graceful API failure handling:** Local SHAP explanations remain available when Groq cannot be accessed.

## System Architecture

```text

CIC-IDS2017 CSV Files

          |

          v

Data Cleaning and Preprocessing

          |

          v

Train/Test Split (80/20)

          |

          v

Feature Selection and StandardScaler

          |

          v

Random Forest Intrusion Detection

          |

          v

Benign / Malicious Prediction

          |

          v

SHAP Feature Explanation

          |

          +--------------------+

          |                    |

          v                    v

Streamlit Dashboard     Groq SOC Analysis

          |                    |

          +--------------------+

          |

          v

Interpretable Security Results

```

## Dataset

The project uses **CIC-IDS2017**, a network intrusion detection dataset containing benign traffic and several attack categories.

For this baseline, the original multiclass labels are converted into binary labels:

- `0` — Benign traffic

- `1` — Malicious traffic

### Preprocessing Summary

| Property | Value |

|---|---:|

| Source CSV files | 8 |

| Original network flows | 2,830,743 |

| Duplicate rows removed | 308,381 |

| Rows removed due to invalid features | 1,564 |

| Usable network flows | 2,520,798 |

| Training samples | 2,016,638 |

| Testing samples | 504,160 |

| Input features | 70 |

| Removed zero-variance features | 8 |

| Training/testing split | 80% / 20% |

| Random seed | 42 |

The preprocessing pipeline removes duplicates, handles invalid numerical values, performs a stratified train/test split, removes constant features based on training data, and fits `StandardScaler` using training data only.

## Machine Learning Model

The baseline classifier is a **Random Forest** implemented using scikit-learn.

### Model Configuration

| Parameter | Value |

|---|---|

| Algorithm | Random Forest |

| Number of trees | 80 |

| Maximum tree depth | 16 |

| Minimum samples per leaf | 2 |

| Class weighting | Balanced |

| Maximum features | Square root |

| Random state | 42 |

| Input features | 70 |

### Evaluation Results

The model was evaluated on 504,160 network flows from the stratified random test split.

| Metric | Score |

|---|---:|

| Accuracy | 99.88% |

| Malicious precision | 99.37% |

| Malicious recall | 99.89% |

| Malicious F1-score | 99.63% |

### Confusion Matrix

| Actual / Predicted | Benign | Malicious |

|---|---:|---:|

| Benign | 418,475 | 537 |

| Malicious | 90 | 85,058 |

The model correctly classified 503,533 of the 504,160 test flows.

**Important:** These results describe performance on the saved CIC-IDS2017 random test split. They should not be interpreted as guaranteed performance on real-world networks.

## Generalization Evaluation: Held-Out Friday Experiment

To evaluate how well the Random Forest model generalizes to unseen attack categories and temporal distribution shifts, we performed a separate **held-out-day generalization experiment**.

The model was trained strictly on traffic from **Monday through Thursday** (1,924,938 flows) and evaluated exclusively on traffic from **Friday** (592,328 flows). All Friday attack types (**DDoS**, **PortScan**, and **Bot**) were completely absent from the Monday–Thursday training data.

### Baseline vs. Held-Out Day Comparison

| Metric | Random Split (Baseline) | Held-Out Friday (Generalization) |
|---|---:|---:|
| **Accuracy** | 99.88% | 76.42% |
| **Malicious Precision** | 99.37% | 99.42% |
| **Malicious Recall** | 99.89% | 36.77% |
| **Malicious F1-Score** | 99.63% | 53.68% |
| **Benign Specificity** | 99.87% | 99.87% |
| **DDoS Recall** | *(in split)* | 62.91% |
| **PortScan Recall** | *(in split)* | 0.43% |
| **Bot Recall** | *(in split)* | 0.00% |

**Key Finding:** While benign traffic remained highly stationary (99.87% specificity with only 470 false alarms out of 372,238 flows), malicious recall dropped significantly on unfamiliar attacks. The model partially detected unseen DDoS attacks (62.91%) due to structural similarities with training DoS floods, but missed virtually all unseen PortScan (0.43%) and Botnet (0.00%) traffic. This demonstrates that standard random-split benchmarks can significantly overestimate an intrusion detection system's readiness for zero-day threats.

For the complete methodology, cross-split duplicate removal, confusion matrices, and feature importances, see the full research report:  
👉 [**AI-NIDS: Evaluating Generalization to Unseen Network Attacks**](docs/held_out_day_evaluation.md)

## Explainable AI with SHAP

AI-NIDS uses **SHAP (SHapley Additive exPlanations)** to explain individual model predictions.

For a selected network flow, the system identifies influential features and displays:

- Feature name

- Original feature value

- SHAP contribution

- Whether the contribution supports or opposes the predicted class

SHAP helps explain the model's behavior. It does not establish that a feature caused an attack.

## Groq-Powered SOC Reports

The system integrates Groq to produce AI-assisted security analysis using the model prediction, predicted-class probability, and selected SHAP feature evidence.

Reports are structured around:

1. Assessment

2. Evidence

3. Limitations

4. Defensive next steps

The reporting instructions discourage unsupported claims about attack types, IP addresses, victims, or indicators of compromise.

**Security note:** A machine learning prediction is not a confirmed security incident. Groq-generated analysis should be reviewed by a human analyst.

The system also supports graceful failure handling. If the API key is missing or the Groq service fails, local SHAP explanations remain available.

## Streamlit Dashboard

The interactive dashboard allows users to explore network intrusion predictions and their explanations.

It includes classification results, prediction statistics, SHAP feature information, and optional AI-assisted SOC reports.

Run the dashboard locally:

```bash

streamlit run app.py

```

The application normally opens at:

`http://localhost:8501`

## Installation

### 1. Clone the Repository

```bash

git clone YOUR_REPOSITORY_URL

cd AI-NIDS

```

Replace `YOUR_REPOSITORY_URL` with the actual GitHub repository URL.

### 2. Create a Virtual Environment

Python 3.11 is recommended.

On Windows PowerShell:

```powershell

py -3.11 -m venv .venv

.\.venv\Scripts\Activate.ps1

```

### 3. Install Dependencies

```bash

python -m pip install -r requirements.txt

```

### 4. Prepare the Dataset

Obtain the CIC-IDS2017 MachineLearningCVE CSV files from an authorized dataset source.

Place the eight CSV files inside:

```text

data/raw/CIC-IDS-2017/CSVs/MachineLearningCSV/MachineLearningCVE/

```

### 5. Run Preprocessing

```bash

python preprocess.py

```

This generates processed training and testing datasets, the feature list, and the fitted scaler.

### 6. Train the Model

```bash

python safe_train.py

```

The training script produces the Random Forest model and evaluation artifacts.

### 7. Configure Groq (Optional)

Create a local `.env` file using `.env.example` as a reference.

Configure the Groq API key and model identifier required by the application.

Never commit real API keys or `.env` files to GitHub.

### 8. Start the Dashboard

```bash

streamlit run app.py

```

## Automated Tests

Run the automated test suite:

```bash

python -m pytest tests/test_ai_nids.py -v

```

The current baseline passes seven tests covering:

- Model artifact availability

- Feature compatibility

- Binary predictions

- Probability validation

- SHAP explanations

- Missing Groq API key handling

- Simulated Groq API failure handling

Latest verified result: **7 passed, 0 failed**.

## Project Structure

```text
AI-NIDS/
├── app.py
├── preprocess.py
├── train_model.py
├── safe_train.py
├── xai_engine.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .env.example
├── .gitignore
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── experiments/
│   ├── held_out_day.py
│   ├── safe_run_held_out_day.py
│   └── results/held_out_day/
├── tests/
│   ├── test_ai_nids.py
│   └── test_held_out_day.py
└── docs/
    └── held_out_day_evaluation.md
```

Some directories contain generated artifacts or large datasets that may not be included in the public repository.

## Limitations

This project is a research and portfolio prototype, not a production-ready intrusion detection system.

Current limitations include:

- **Random flow splitting:** Related network sessions may appear in both training and testing data, potentially inflating evaluation scores.

- **Dataset generalization:** Performance on CIC-IDS2017 does not establish performance on newer attacks or independent networks.

- **Binary classification:** The model predicts benign or malicious traffic but does not reliably identify a specific attack category.

- **False positives and false negatives:** The classifier can incorrectly flag benign traffic or miss malicious flows.

- **SHAP interpretation:** Feature contributions explain model predictions, not attack causality.

- **AI-generated analysis:** Groq reports may contain inaccuracies and require analyst review.

- **Deployment:** The current application is designed for local experimentation and has not been validated as a production network monitoring system.

## Future Improvements

Potential improvements include:

- Evaluation using separate capture days and independent network datasets

- Multiclass attack classification

- Additional model comparisons

- Improved SOC report validation

- Container security hardening and automated deployment

- More comprehensive automated testing

- Better monitoring and reproducibility documentation

## Technology Stack

**Language:** Python 3.11

**Machine Learning:** scikit-learn, pandas, NumPy

**Explainability:** SHAP

**Generative AI:** Groq API

**Dashboard:** Streamlit

**Testing:** pytest

**Containerization:** Docker

## Docker Deployment

AI-NIDS supports Docker deployment using Python 3.11 and Streamlit. The container runs the dashboard, Random Forest predictions, SHAP explanations, and optional Groq SOC Analyst reports.

### Build the Docker Image

From the project root, run:

```powershell

docker build -t ai-nids:1.0 .

```

### Run the Docker Container

Before starting, ensure the trained model files are available in `models/` and the processed test data is available in `data/processed/`.

Create a local `.env` file using `.env.example` as a reference. Add your own Groq API key. Never commit `.env` to GitHub.

On Windows PowerShell, run:

```powershell

docker run -d `

  --name ai-nids-app `

  -p 8502:8501 `

  --env-file .env `

  --mount "type=bind,source=${PWD}\models,target=/app/models,readonly" `

  --mount "type=bind,source=${PWD}\data\processed,target=/app/data/processed,readonly" `

  ai-nids:1.0

```

Open the dashboard at:

http://localhost:8502

### Verify the Container

```powershell

docker ps -a --filter "name=ai-nids-app"

```

### Stop and Remove the Container

```powershell

docker rm -f ai-nids-app

```

### Deployment Notes

- Docker uses the existing trained model; retraining is not required.

- Models and processed datasets are mounted read-only and are not included in the image.

- Groq is optional and requires an API key supplied at runtime.

- The Docker deployment has been locally tested with Streamlit, Random Forest predictions, SHAP explanations, and Groq reports.

- This deployment is intended for local demonstration and experimentation, not production network monitoring.

## Project Purpose

AI-NIDS was developed to explore how machine learning, explainable AI, and large language models can work together in cybersecurity.

The goal is not only to classify network traffic but also to make predictions understandable and provide useful evidence for further security investigation.

## Project Credits

**Project:** AI-NIDS — Explainable AI Network Intrusion Detection System

**Developed by:** Syed Muhammad Huzaifa Chishty

**Role:** Founder & AI Engineer

**Organization:** SASH Innoventures

AI-NIDS is an applied AI and cybersecurity project developed as part of the founder's technical portfolio at SASH Innoventures.

The project combines machine learning, explainable AI (SHAP), and LLM-assisted security analysis to demonstrate an interpretable approach to network intrusion detection.
