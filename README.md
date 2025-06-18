# Generartive-AI-Knee-Injury-predictor-
ACL Reinjury Risk Prediction Pipeline

A simple, reproducible Python project to predict the risk of reinjury after ACL reconstruction. It reads patient data from Excel, encodes clinical features, applies PCA for visualization, and computes a weighted risk score.

Repository Structure

headache.py (a.k.a. head2.py): Main script for single-patient risk prediction.

head2_multi.py: Script to process multiple new patients from a CSV file.

run_headache.sh: Launcher script that handles virtual environment creation, dependency installation, and runs head2.py.

data collection.xlsx: Baseline dataset of existing patients (47 rows).

new_patients.csv: Template CSV for multiple new patient inputs.

requirements.txt: (Optional) List of Python libraries.

README.md: This documentation file.

Prerequisites

macOS / Linux with Python 3.9+ installed (Homebrew Python will work).

Git (if cloning from GitHub).

Basic familiarity with the Terminal or command prompt.

Setup & Installation

Clone the repo
git clone https://github.com/yourusername/acl-reinjury-risk.git
cd acl-reinjury-risk
Create a virtual environment
python3 -m venv venv
Activate the environment
source venv/bin/activate
Install dependencies
Usage

1) Single-Patient Prediction (headache.py or head2.py)

Run with patient details as flags
pip install pandas scikit-learn matplotlib openpyxl
python3 headache.py \
  --age 22 \
  --gender Male \
  --bmi 25 \
  --activity Competitive \
  --graft Allograft \
  --muscles "Peroneus longus" \
  --diameter 7 \
  --femoral Suspension \
  --tibial Expansion \
  --rehab Conventional \
  --mechanism Pivoting \
  --beighton 6 \
  --mmq 2 \
  --mmh 2
  This will:

Print the patient’s weighted risk score and prediction.

Pop up a PCA plot (risk_map.png) showing existing patients (blue) and the new patient (red).

Save existing_patients_results.csv and new_patient_prediction.csv in the working folder.

2) Multi-Patient Prediction (head2_multi.py)

Prepare new_patients.csv with header:
Age,Gender,BMI,Activity level,Graft type,Muscles used for Graft,Diameter of graft,Femoral fixation,Tibial fixation,Rehabilitation Protocol,Mechanism of injury,Beighton's score,MMT Quadriceps,MMT Hamstrings
Add one row per patient (see new_patients.csv template).

Run:
python3 head2_multi.py --newcsv new_patients.csv
Outputs:

Console: Each new patient’s risk score & prediction.

risk_map_multi.png: PCA plot with all new patients in red.

existing_patients_results.csv & new_patients_predictions.csv with PCA coordinates.

3) Launcher Script (run_headache.sh)

To automate setup + single-patient run:
chmod +x run_headache.sh
./run_headache.sh --age 22 --gender Male --bmi 25 --activity Competitive ...
Creates/activates venv

Installs missing packages on first run

Runs head2.py with passed flags

Input Data Format

data collection.xlsx: Excel file of 47 existing patients. Columns should include:

Age (e.g. "26Y")

Gender, BMI, Activity level, Graft type, Muscles used for Graft, Diameter of graft, Femoral fixation, Tibial fixation, Rehabilitation Protocol, Mechanism of injury, Beighton's score, MMT Quadriceps, MMT Hamstrings

new_patients.csv: CSV for multiple inputs with matching column headers (see Usage section).

Outputs

existing_patients_results.csv: PCA coordinates (PC1, PC2) + risk score for baseline patients.

new_patient_prediction.csv or new_patients_predictions.csv: PCA coords + risk score + "High"/"Lower" label for new patients.

risk_map.png or risk_map_multi.png: PCA scatterplot image.

How It Works (Pipeline Overview)

Load Excel → drop irrelevant columns → fill blanks.

Encode features: Age, gender, BMI, graft details, muscle strength flags, etc.

Impute missing flags with column medians.

Scale features (zero mean, unit variance).

PCA: project 15 features down to 2 dimensions for visualization.

Risk scoring: sum weighted flags (e.g. small graft = 2.0, weak muscles = 1.5).

Overlay new patients onto the same PCA space.

Save results (CSV + PNG).

Customization

Weights: Adjust the weights dictionary in headache.py to change how much each feature contributes to risk.

Threshold: Modify threshold = 4.0 in main() to tune the High/Lower decision.

Feature logic: Edit encode_* functions to support new categories or different cut-offs.
