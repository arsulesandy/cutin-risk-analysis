# Cut-in Risk Analysis

This repository contains the code developed as part of a Master’s thesis on **cut-in lane-change risk analysis** using naturalistic highway trajectory data.

The project focuses on detecting cut-in scenarios in a **clear and reproducible way** and analysing how different risk indicators behave during these manoeuvres. The work is intended for **offline analysis and research use**, not for real-time driving systems or controllers.

The initial implementation is based on the **highD dataset**, but the codebase is structured to allow additional trajectory datasets to be integrated later without changing the core analysis logic.

---

## Scope of the project

The main objectives of this project are:

- To detect lane-change events from trajectory data using explicit, rule-based logic
- To identify cut-in scenarios based on vehicle neighbour relations
- To compute established surrogate risk indicators as time series
- To compare indicators in terms of timing, severity, and agreement
- To provide a maintainable and reproducible analysis pipeline

The goal is not to define a universal standard for cut-in risk, but to make assumptions, definitions, and results transparent and repeatable.

---

## What the pipeline does

The analysis pipeline is organized into four main stages:

### 1. Dataset preparation
- Load vehicle trajectories and metadata
- Normalize time, units, and coordinate conventions
- Validate lane assignments and neighbour relationships
- Perform basic data quality checks

### 2. Scenario detection
- Detect lane-change events based on lane assignment over time
- Identify cut-in situations when a lane-changing vehicle becomes the immediate predecessor of another vehicle in the target lane
- Extract consistent time windows before, during, and after the manoeuvre

### 3. Risk indicator computation
For each cut-in scenario, the pipeline computes several established indicators, including:
- Time-to-Collision (TTC)
- Time headway
- Braking-demand style indicators
- Safe-gap based measures

Indicators are computed as time series to capture how risk evolves over the manoeuvre.

### 4. Analysis and inspection
- Compare indicator timing and severity
- Analyse agreement and disagreement between indicators
- Generate summary statistics and figures
- Visually inspect selected scenarios for sanity checking

---

## Supported datasets

### highD (primary dataset)

The project currently supports the highD dataset, which consists of drone-recorded highway vehicle trajectories with detailed neighbour information.

Due to licensing restrictions, **raw highD data is not included** in this repository. You must obtain access separately and store the data locally.

### Extending to other datasets

The codebase is designed to be dataset-agnostic. Support for additional trajectory datasets can be added by implementing a dataset adapter, without modifying the detection or analysis logic.

---

## Repository structure (overview)

- `src/cutin_risk/`  
  Core library code, including dataset adapters, detection logic, risk indicators, analysis utilities, and visualization tools.

- `configs/`  
  Configuration files for datasets and experiments. All runs are driven by configuration to ensure reproducibility.

- `data/`  
  Local data directories. Raw data is gitignored. Only derived or intermediate artifacts should be stored here.

- `outputs/`  
  Generated results such as extracted scenarios, figures, and tables. Outputs are organized per run.

- `tests/`  
  Unit and integration tests, primarily focused on indicator calculations and detection logic.

- `notebooks/`  
  Exploration and sanity-check notebooks. These are not part of the main pipeline.

- `docs/`  
  Design notes, methodological explanations, and project decisions.

---

## Running the code

### Environment setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
