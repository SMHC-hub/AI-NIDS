# AI-NIDS: Evaluating Generalization to Unseen Network Attacks

**Syed Muhammad Huzaifa Chishty**  
Founder & AI Engineer, SASH Innoventures  

---

## 1. Executive Summary

Machine learning models for Network Intrusion Detection Systems (NIDS) frequently achieve reported accuracy and F1-scores exceeding 99% on benchmark datasets when evaluated with standard stratified random splitting. In practical network environments, however, network defenders observe that classifiers frequently degrade when encountering unfamiliar traffic patterns, new attack variants, or shifted operational baselines.

To examine this performance disparity, we conducted a temporal held-out generalization experiment using the CIC-IDS2017 dataset within the AI-NIDS project. Rather than randomly distributing network flows across training and test splits, we trained a Random Forest binary classifier strictly on traffic captured from **Monday through Thursday** (1,924,938 clean flows) and evaluated it exclusively on traffic captured on **Friday** (592,328 clean flows).

Crucially, in the CIC-IDS2017 capture schedule, all malicious attack categories executed on Friday—specifically **DDoS**, **PortScan**, and **Botnet** traffic—were completely absent during Monday through Thursday. Consequently, this setup serves as an empirical evaluation of out-of-distribution generalization to unseen attack categories, rather than an in-distribution evaluation.

### Key Findings

- **Benign Baseline Stability:** The classifier retained a **99.87% specificity** on Friday benign traffic, generating only 470 false alarms out of 372,238 benign flows (a 0.13% false positive rate).
- **Partial Transfer on Volumetric Attacks:** The model detected **62.91%** of unseen Friday **DDoS** flows (80,530 out of 128,012), likely because structural similarities with Monday–Wednesday Denial-of-Service attacks (such as DoS Hulk and DoS GoldenEye) enabled partial feature transfer.
- **Reconnaissance and Stealth Blindspots:** The model identified only **0.43%** of **PortScan** flows (387 out of 90,130) and **0.00%** of **Botnet** flows (0 out of 1,948), missing virtually all scanning probes and command-and-control heartbeats.
- **Overall Metric Shift:** Under held-out Friday traffic, malicious detection recall fell from **99.89%** (random split) to **36.77%**, bringing the malicious F1-score from **99.63%** down to **53.68%** (with an overall accuracy of **76.42%**).

These results illustrate that random-split benchmarks can significantly overstate an intrusion detection model's ability to recognize novel attack behavior.

---

## 2. Research Question

> **Primary Research Question:**  
> Can a supervised Random Forest classifier trained exclusively on Monday–Thursday network flows from CIC-IDS2017 reliably identify malicious flows on Friday, when the test traffic contains attack categories completely absent from the training set?

This question investigates whether decision boundaries formed over specific attack types (brute-force password guessing, web application attacks, infiltration, and several DoS variants) extract generalizable indicators of maliciousness, or whether they primarily memorize the statistical artifacts of the attack categories present during training.

---

## 3. Dataset and Experimental Setup

### 3.1 The CIC-IDS2017 Dataset

The Canadian Institute for Cybersecurity (CIC) captured the CIC-IDS2017 dataset across five consecutive weekdays, generating eight CSV files containing bidirectional flow statistics extracted by CICFlowMeter. 

We partitioned the eight original raw CSV files strictly by capture day:

| Capture Day | File Name | Raw Rows | Assigned Split |
| :--- | :--- | :---: | :---: |
| **Monday** | `Monday-WorkingHours.pcap_ISCX.csv` | 529,918 | **Training** |
| **Tuesday** | `Tuesday-WorkingHours.pcap_ISCX.csv` | 445,909 | **Training** |
| **Wednesday** | `Wednesday-workingHours.pcap_ISCX.csv` | 692,703 | **Training** |
| **Thursday** | `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 170,366 | **Training** |
| **Thursday** | `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 288,602 | **Training** |
| **Friday** | `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 191,033 | **Testing** |
| **Friday** | `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 286,467 | **Testing** |
| **Friday** | `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 225,745 | **Testing** |

### 3.2 Label Formulation

This experiment evaluates binary network intrusion detection:
- **Class 0 (BENIGN):** Normal background network activity.
- **Class 1 (MALICIOUS):** Network flows generated during simulated attacks.

Multiclass attack labels were preserved during ingestion to enable per-attack evaluation on the test set, but mapped to `1` during model fitting.

### 3.3 Attack Distribution Across Splits

The table below contrasts the attack classes available during training versus those evaluated during testing:

| Attack Category | Monday–Thursday (Training) | Friday (Held-Out Test) | Status in Test Evaluation |
| :--- | :---: | :---: | :---: |
| **BENIGN** | 1,719,853 | 372,238 | Seen during training |
| **DoS Hulk** | 172,846 | 0 | Training only |
| **DoS GoldenEye** | 10,286 | 0 | Training only |
| **FTP-Patator** | 5,931 | 0 | Training only |
| **DoS slowloris** | 5,385 | 0 | Training only |
| **DoS Slowhttptest** | 5,228 | 0 | Training only |
| **SSH-Patator** | 3,219 | 0 | Training only |
| **Web Attack – Brute Force** | 1,470 | 0 | Training only |
| **Web Attack – XSS** | 652 | 0 | Training only |
| **Infiltration** | 36 | 0 | Training only |
| **Web Attack – SQL Injection** | 21 | 0 | Training only |
| **Heartbleed** | 11 | 0 | Training only |
| **DDoS** | 0 | 128,012 | **Unseen attack category** |
| **PortScan** | 0 | 90,130 | **Unseen attack category** |
| **Bot** | 0 | 1,948 | **Unseen attack category** |

### 3.4 Preprocessing and Cleaning

The pipeline applies the same cleaning criteria as the baseline AI-NIDS project:
1. **Schema Validation:** Strips trailing whitespace from column headers and verifies schema consistency across chunks.
2. **Internal Deduplication:** Removes duplicate rows within each partition (201,442 rows in training; 87,351 rows in test).
3. **Invalid Value Removal:** Converts non-numeric strings to `NaN`, replaces `+inf` and `-inf` with `NaN`, and discards rows containing non-finite values (1,118 rows in training; 461 rows in test).
4. **Data Typing:** Downcasts numeric features to `float32` to remain within memory constraints.

### 3.5 Random Forest Configuration

The classifier uses the exact configuration of the AI-NIDS baseline model:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `n_estimators` | 80 | Number of decision trees |
| `max_depth` | 16 | Maximum tree depth |
| `min_samples_leaf` | 2 | Minimum leaf sample threshold |
| `max_features` | `'sqrt'` | Features considered per split ($\approx \sqrt{70} \approx 8$) |
| `class_weight` | `'balanced'` | Adjusts tree weights inversely proportional to class frequencies |
| `n_jobs` | 4 | Parallel CPU workers |
| `random_state` | 42 | Deterministic seed |

---

## 4. Data Leakage Prevention

To ensure rigorous evaluation, we applied several isolation controls:

### 4.1 Strict Split-Isolated Feature Selection and Scaling

- **Zero-Variance Filtering:** Eight columns exhibited zero variance in the Monday–Thursday training split:
  `['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']`.
  These columns were dropped from both splits based strictly on training calculations. The remaining 70 features were retained.
- **Feature Scaling:** `StandardScaler` was fitted strictly on `x_train`. Friday data was normalized using the training mean and standard deviation:
  $$\mu_{\text{scaler}} = \mu_{\text{train}}, \quad \sigma_{\text{scaler}} = \sigma_{\text{train}}$$
  Automated test assertions in [`tests/test_held_out_day.py`](file:///f:/AI-NIDS/tests/test_held_out_day.py) verify that scaler parameters match training data within a numerical tolerance of $10^{-4}$ and reject any test data inclusion.

### 4.2 Cross-Split Duplicate Removal

Network traffic often contains identical baseline flows across different days (e.g., automated NTP updates, DNS handshakes, or idle TCP connections). If an identical flow appears in both the training set and the test set, testing on that flow can constitute subtle information leakage.

Using vectorized 64-bit row hashing followed by exact candidate confirmation:
- We scanned all 615,433 clean Friday test flows against the 1,924,938 clean Monday–Thursday training flows.
- Exactly **23,105 identical flow records** were identified in the Friday test set that matched records in the training set.
- All 23,105 overlapping records were **purged from the Friday test set**, leaving **592,328 unique test flows**.

### 4.3 Safeguard Limitations

While cross-split deduplication removes identical feature vectors, it cannot eliminate all forms of correlation across days:
- Identical victim host IP configurations, operating system TCP/IP stack implementations, and fixed network topologies remain shared between days in the simulated testbed.
- Even with identical records removed, benign services (such as internal web servers) may produce closely clustered feature vectors across days.

---

## 5. Baseline vs. Held-Out Evaluation

The table below contrasts the baseline random-split evaluation against the held-out Friday evaluation:

| Metric | Baseline Model (Random Stratified Split) | Held-Out Model (Monday–Thursday $\to$ Friday) | Difference |
| :--- | :---: | :---: | :---: |
| **Evaluation Type** | Random 80/20 Stratified Flow Split | Temporal Held-Out Capture Day | Structural Shift |
| **Evaluated Test Rows** | 504,160 | 592,328 | +88,168 |
| **Input Features** | 70 | 70 | Identical |
| **Overall Accuracy** | **99.88%** (`0.998756`) | **76.42%** (`0.764247`) | -23.46% |
| **Malicious Precision** | **99.37%** (`0.993726`) | **99.42%** (`0.994225`) | +0.05% |
| **Malicious Recall** | **99.89%** (`0.998943`) | **36.77%** (`0.367654`) | **-63.12%** |
| **Malicious F1-Score** | **99.63%** (`0.996328`) | **53.68%** (`0.536804`) | **-45.95%** |
| **Macro Average F1** | 99.79% | 68.93% | -30.86% |

> **Important Interpretation Note:**  
> These numbers do not reflect an algorithmic comparison between two classifiers. Both models use the same Random Forest implementation and hyperparameters. Instead, the score differential reflects the difference between **in-distribution interpolation** (where the test set shares attack sessions with the training set) and **out-of-distribution generalization** (where the test set contains entirely novel attack classes).

---

## 6. Confusion Matrix

On the held-out Friday test set (592,328 flows), the confusion matrix counts are:

| Actual \ Predicted | Predicted Benign (`0`) | Predicted Malicious (`1`) | Total Actual | Class Metric |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Benign (`0`)** | **371,768** (TN) | **470** (FP) | 372,238 | Specificity: **99.87%** |
| **Actual Malicious (`1`)** | **139,173** (FN) | **80,917** (TP) | 220,090 | Recall: **36.77%** |
| **Total Predicted** | 510,941 | 81,387 | **592,328** | Precision: **99.42%** |

### Analysis of Errors

1. **Low False Alarm Rate (470 FPs):**  
   Out of 372,238 benign flows, only 470 were falsely classified as malicious, representing a False Positive Rate (FPR) of **0.13%**. This yields a high precision of **99.42%** when the classifier does issue an alert.
2. **High False Negative Rate (139,173 FNs):**  
   The primary failure mode is missed attacks. The model failed to detect 139,173 out of 220,090 malicious flows (a false negative rate of **63.23%**), driven almost entirely by undetected scanning and botnet traffic.

---

## 7. Per-Attack Results

Evaluating the binary predictions against the underlying multiclass labels reveals the breakdown across attack categories:

| Traffic Label | Ground Truth Category | Unseen in Training? | Support Count | Detected as Malicious | Misclassified as Benign | Detection Metric | Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **BENIGN** | Benign | No | 372,238 | 470 (FP) | 371,768 (TN) | **Specificity** | **99.87%** |
| **DDoS** | Malicious | **Yes** | 128,012 | 80,530 (TP) | 47,482 (FN) | **Detection Recall** | **62.91%** |
| **PortScan** | Malicious | **Yes** | 90,130 | 387 (TP) | 89,743 (FN) | **Detection Recall** | **0.43%** |
| **Bot** | Malicious | **Yes** | 1,948 | 0 (TP) | 1,948 (FN) | **Detection Recall** | **0.00%** |

```text
Detection Recall by Attack Category (Friday Held-Out):
=============================================================================
DDoS       [===============================                             ] 62.91% (80,530 / 128,012)
PortScan   [                                                            ]  0.43% (387 / 90,130)
Bot        [                                                            ]  0.00% (0 / 1,948)
BENIGN*    [============================================================] 99.87% (371,768 / 372,238)
=============================================================================
* BENIGN metric is specificity (benign flows correctly identified as benign).
```

---

## 8. Feature Importance

The top 15 global features derived from the Monday–Thursday Random Forest model (based on Mean Decrease in Impurity / Gini importance) are:

| Rank | Feature Name | Gini Importance | Cumulative Importance |
| :---: | :--- | :---: | :---: |
| 1 | `Bwd Packet Length Std` | 0.1023 | 10.23% |
| 2 | `Bwd Packet Length Mean` | 0.0976 | 20.00% |
| 3 | `Max Packet Length` | 0.0702 | 27.02% |
| 4 | `Average Packet Size` | 0.0593 | 32.94% |
| 5 | `Bwd Packet Length Max` | 0.0558 | 38.53% |
| 6 | `Packet Length Mean` | 0.0527 | 43.79% |
| 7 | `Total Length of Bwd Packets` | 0.0506 | 48.85% |
| 8 | `Destination Port` | 0.0474 | 53.59% |
| 9 | `Packet Length Variance` | 0.0454 | 58.13% |
| 10 | `Avg Bwd Segment Size` | 0.0446 | 62.58% |
| 11 | `Packet Length Std` | 0.0331 | 65.89% |
| 12 | `Flow IAT Mean` | 0.0220 | 68.09% |
| 13 | `Fwd IAT Min` | 0.0174 | 69.83% |
| 14 | `Fwd Header Length` | 0.0165 | 71.48% |
| 15 | `Bwd Packets/s` | 0.0163 | 73.11% |

### Methodological Caveat on Feature Importance

Gini-based feature importances reflect how frequently a feature is selected for splits and the reduction in node impurity it achieves on the **training set**. They describe model mechanics rather than physical causality. For example, high backward packet length metrics reflect the asymmetric response sizes typical of HTTP DoS and file downloads, but their prominence does not imply that packet length alone causes traffic to be malicious.

---

## 9. Discussion

### 9.1 Why Did DDoS Achieve 62.91% Recall?

Monday through Thursday training data included extensive Denial-of-Service traffic:
- `DoS Hulk` (172,846 flows)
- `DoS GoldenEye` (10,286 flows)
- `DoS slowloris` (5,385 flows)
- `DoS Slowhttptest` (5,228 flows)

These training attacks established decision splits centered on high packet counts, elevated backward packet lengths, and specific inter-arrival timing patterns. When Friday DDoS traffic was evaluated, the high-volume flood characteristics partially satisfied these existing DoS decision boundaries, enabling **62.91%** of DDoS flows to be flagged. The remaining 37.09% of missed DDoS flows likely differed in flow duration, packet distribution, or protocol header flags from the HTTP-based DoS variants seen in training.

### 9.2 Why Did PortScan Achieve Only 0.43% Recall?

Port scanning represents reconnaissance rather than exploitation:
- Port scans typically generate short, lightweight flows consisting of a single TCP SYN packet, an immediate RST/ACK response, or a connection timeout.
- Monday through Thursday training contained **zero port scanning traffic**. The malicious training examples consisted of high-volume web attacks, brute force authentication, and volumetric DoS floods.
- As a consequence, the classifier learned that malicious flows generally feature multiple packets, bidirectional data transfers, or elevated payload lengths. Because port scan flows resemble benign short connections (such as closed port probes or DNS lookups), the model classified 89,743 out of 90,130 port scan flows as benign.

### 9.3 Why Did Bot Traffic Achieve 0.00% Recall?

The Friday Botnet traffic (ARES botnet) consists of periodic command-and-control (C2) communications and heartbeats:
- These flows are low-rate, small in packet size, and designed to blend into normal HTTP/HTTPS web browsing.
- Because the training set contained no command-and-control behavior, the model possessed no decision splits sensitive to low-frequency beaconing.
- All 1,948 bot flows were classified as benign.

---

## 10. Limitations

1. **Unseen Attack Categories vs. Temporal Drift:**  
   Because Friday introduced three completely new attack types, this experiment conflates two distinct forms of distribution shift: temporal drift (traffic shifting from day to day) and unseen class generalization (evaluating attack families never shown during training).
2. **Benchmark Testbed Artifacts:**  
   CIC-IDS2017 was generated in a simulated laboratory testbed over a single week in 2017. Network topologies, background traffic generators (B-Profile), and attack scripts do not reflect modern enterprise networks, encrypted protocols (TLS 1.3, QUIC), or current threat actor tradecraft.
3. **Optimism of Random Splitting:**  
   The contrast between 99.88% accuracy (random split) and 76.42% accuracy (held-out day) demonstrates that standard random splitting allows classifiers to memorize session-level flow identifiers. Random-split metrics should be treated as upper-bound memorization tests rather than deployment-ready evaluations.
4. **Binary Formulation Constraints:**  
   A binary model cannot provide actionable attribution to security analysts. Even when a flow is correctly flagged (such as DDoS), the system does not identify the attack type or the underlying threat mechanism.
5. **Heuristic Duplicate Filtering:**  
   Exact row deduplication removes identical feature vectors but does not address near-identical flows differing only by a minor timestamp or ephemeral port variation.
6. **Absence of Real-World Deployment:**  
   This evaluation was conducted offline on static CSV files. Real-world deployment involves live packet capture, stream-based flow reassembly, packet loss, and dynamic network routing, none of which are tested here.

---

## 11. Future Work

1. **Multi-Day Cross-Validation:**  
   Evaluate rotation-based held-out days (e.g., hold out Wednesday while training on Monday, Tuesday, Thursday, and Friday) to evaluate whether web attacks and brute-force attacks exhibit reciprocal generalization.
2. **Model Architecture Comparisons:**  
   Compare the Random Forest against alternative paradigms:
   - Deep learning architectures (MLP, TabNet, Temporal Convolutional Networks).
   - Semi-supervised and unsupervised anomaly detectors (Isolation Forests, One-Class SVMs, Autoencoders) designed specifically for out-of-distribution discovery.
3. **Threshold Calibration:**  
   Analyze Precision-Recall trade-offs by lowering the classification decision threshold from 0.50 to evaluate whether PortScan and Bot recall can be improved without unacceptable increases in false alarms.
4. **Cross-Dataset Validation:**  
   Evaluate models trained on CIC-IDS2017 on completely separate intrusion datasets, such as CSE-CIC-IDS2018, UNSW-NB15, or TON_IoT.

---

## 12. Reproducibility

### Environment Requirements

- Python 3.11
- Dependencies specified in `requirements.txt` (scikit-learn, pandas, numpy, joblib, psutil)
- Raw CIC-IDS2017 MachineLearningCVE CSV files located in:
  `data/raw/CIC-IDS-2017/CSVs/MachineLearningCSV/MachineLearningCVE/`

### Running the Experiment

To execute the experiment with real-time memory monitoring and an 11.0 GiB safety limit:

```powershell
# From the project root on Windows PowerShell:
.\.venv\Scripts\python.exe experiments/safe_run_held_out_day.py
```

Alternatively, run the script directly:

```powershell
.\.venv\Scripts\python.exe experiments/held_out_day.py
```

### Generated Artifacts

All experiment artifacts are saved under `experiments/results/held_out_day/`:

- [`held_out_day_metrics.json`](file:///f:/AI-NIDS/experiments/results/held_out_day/held_out_day_metrics.json) — Full structured JSON containing overall metrics, classification report, class counts, and split statistics.
- [`per_attack_recall.csv`](file:///f:/AI-NIDS/experiments/results/held_out_day/per_attack_recall.csv) — Detailed per-attack support counts and detection recalls.
- [`feature_importances.csv`](file:///f:/AI-NIDS/experiments/results/held_out_day/feature_importances.csv) — Ranked Gini feature importances.
- [`rf_held_out_day_model.pkl`](file:///f:/AI-NIDS/experiments/results/held_out_day/rf_held_out_day_model.pkl) — Serialized Random Forest model weights.
- [`scaler.pkl`](file:///f:/AI-NIDS/experiments/results/held_out_day/scaler.pkl) — Fitted StandardScaler.
- [`feature_columns.pkl`](file:///f:/AI-NIDS/experiments/results/held_out_day/feature_columns.pkl) — Retained feature column list.

---

## 13. Conclusion

The held-out-day generalization experiment provides an empirical demonstration of the gap between random-split validation and out-of-distribution testing in machine learning for network security.

When evaluated strictly on Friday traffic, the Random Forest model maintained strong benign specificity (**99.87%**, 470 false alarms out of 372,238 benign flows) and demonstrated moderate transfer to volumetric DDoS traffic (**62.91% recall**). However, it failed to detect reconnaissance scanning (**0.43% recall**) and botnet command-and-control activity (**0.00% recall**), which had no structural analogues in the training set.

These findings highlight that high random-split benchmark scores must not be mistaken for robust defense against novel threats. Supervised models trained on historical network flows excel at recognizing familiar attack patterns, but effective defense against unseen attack categories requires combining supervised models with anomaly detection, signature-based verification, and human-in-the-loop analysis.

---

*Report prepared as part of the AI-NIDS project.*  
*Author: Syed Muhammad Huzaifa Chishty — Founder & AI Engineer, SASH Innoventures*
