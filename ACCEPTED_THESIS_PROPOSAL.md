# A Framework for Cut-In Vehicle Detection and Lane-Change Risk Analysis in Naturalistic Highway Trajectories

---

## Student Information

**Name 1:**  
Sandeep Arsule

**Completed courses relevant for thesis work:**
- Decision-making for autonomous systems (SSY236)
- Advanced simulation and machine learning (TIF345)
- Advanced Software Engineering for AI/ML-Enabled Systems (DAT550)
- Empirical software engineering (DAT246)
- Advanced requirements engineering (DAT231)
- Quality assurance and testing (DAT321)

**Name 2:**  
Shradha Shinde

**Completed courses relevant for thesis work:**
- Advanced Software Engineering for AI/ML-Enabled Systems (DAT550)
- Advanced requirements engineering (DAT231)
- Empirical software engineering (DAT246)
- Software evolution project (DAT266)

---

## Requested Information

- **Supervisor:** Claude Gu Jun-Yi
- **Examiner:** Aris Alissandrakis
- **Company involvement:** None
- **Company contact:** Not applicable
- **Web form ID:** 7819

---

## Relation to Previous Project Work

This thesis does not continue or build on an Industrial Practice Project or a Research Project course. It is planned and executed as an independent thesis project.

---

## Introduction

Lane changes are routine on multilane highways, but a subset occurs with small gaps and high relative speed. Such interactions, often perceived as cut-ins, can force abrupt braking and contribute to rear-end and lateral conflicts. For safety-oriented driver assistance and automated driving, it is important to (i) detect these scenarios consistently in real traffic data and (ii) quantify how risk develops during the manoeuvre in a reproducible and interpretable way.

Large-scale naturalistic trajectory datasets enable this type of analysis. The highD dataset provides high-precision vehicle trajectories and surrounding-vehicle relations from drone recordings on German highways, making large-scale scenario extraction feasible [highd_2018]. However, practical analysis is challenged by inconsistent cut-in definitions, unlabelled interactions, and differences across risk indicators, especially in high-speed traffic where available reaction time is limited.

Prior work includes surrogate safety measures such as Time-to-Collision (TTC) [hayward1972], practical indicator suites such as SSAM [ssam2008], and more recent reviews and validation studies that discuss strengths, limitations, and applicability of surrogate safety measures in modern safety modelling [wang2021ssm, johnsson2021validation]. Recent research also predicts lane-change risk using highD-derived samples and interaction features [shangguan2022, liu2025multitask]. In addition, automated cut-in and lane-merging planning uses explicit phase structure and risk evaluation, for example the finite-state-machine formulation with TTC-based risk handling by Hwang et al. [hwang2022fsmrl]. While that work targets planning and control, it motivates clear phase definitions and explicit risk logic, which are also relevant for offline scenario segmentation and analysis.

From our perspective as software engineering students, we also want the thesis outcome to be reusable beyond one-off experimentation. Therefore, the planned result is not only an analysis report, but also a maintainable and well-documented framework that can reproduce scenario extraction, metrics, and figures from configuration.

This thesis develops a modular framework that detects cut-in scenarios in naturalistic highway trajectory data and performs a systematic comparison of risk indicators on the extracted events.

---

## Statement of the Problem

Detecting safety-relevant cut-ins in naturalistic datasets is challenging because:

- Naturalistic datasets contain large volumes of unlabelled interactions, and definitions of "cut-in" differ across studies.
- Risk indicators capture different aspects of danger and may trigger at different times during a manoeuvre.
- Threshold-based logic can be sensitive to traffic state, speed level, and modelling assumptions.
- A practical framework must consider the trade-off between detection quality and processing efficiency when analysing large datasets.

The problem addressed in this thesis is the lack of an integrated, data-driven pipeline that:

- extracts cut-in scenarios from trajectory data using explicit and reproducible rules based on lane assignment and neighbour relations,
- computes a comparable set of established risk indicators for each extracted scenario,
- evaluates and contrasts indicator behaviour (timing, severity ranking, agreement/disagreement) under a consistent protocol.

---

## Purpose of the Study

The purpose of this study is to design, implement, and evaluate a modular framework for cut-in detection and lane-change risk analysis in highway trajectory datasets. The framework will:

- follow a clear module structure for *detection*, *post-processing*, and *analysis*,
- detect lane changes and identify cut-in interactions using lane assignment and neighbour relations,
- extract time-aligned scenario segments (before, during, and after the lane change),
- compute kinematic and interaction features (relative distance, relative speed, headway, acceleration profiles),
- implement and compare established risk indicators (e.g., TTC, time headway, braking-demand style indicators, safe-gap measures),
- evaluate detection consistency and indicator usefulness at scale, including computational cost.

In addition, the implementation will emphasize reproducibility and maintainability through configuration-driven runs, versioned outputs, and basic automated tests for core metric computations.

---

## Review of the Literature

### Surrogate safety measures

TTC is a widely used conflict indicator based on time remaining to collision under simplifying motion assumptions [hayward1972]. SSAM operationalizes several surrogate measures for conflict analysis [ssam2008]. More recent work reviews surrogate safety measures in modern safety modelling (including connected and automated vehicles) and discusses their use and limitations in practice [wang2021ssm]. Validation-oriented studies also highlight how surrogate measures behave under different traffic conditions and modelling assumptions [johnsson2021validation].

### Cut-in and lane-merging algorithms with explicit risk logic

Recent automated driving work proposes structured, phase-based cut-in strategies with explicit risk evaluation. For example, Hwang et al. model lane-merging cut-ins using a finite-state machine with phases (preparation, approach, negotiation, execution) and TTC-based risk logic to manage transitions [hwang2022fsmrl]. Even though this thesis does not implement a controller, this structured view is relevant for how scenarios can be segmented and how risk can be interpreted across manoeuvre phases.

### Trajectory datasets and dataset constraints

The highD dataset enables large-scale extraction of lane changes and surrounding-vehicle relations [highd_2018]. At the same time, the limited road context and short recording segments require careful assumptions when interpreting lane-change intent [wanggao2025highd].

### Data-driven lane-change risk modelling

Recent work increasingly predicts lane-change risk using highD-derived samples and interaction features, for example by combining intention recognition with risk prediction [shangguan2022] or by multi-task learning for manoeuvre and risk prediction [liu2025multitask]. These works reinforce the need for consistent scenario extraction and transparent risk definitions, which this thesis addresses from an offline framework perspective.

---

## Research Questions

1. How can cut-in scenarios be detected reproducibly using lane assignment and neighbour relations in a naturalistic highway dataset?
2. Which features best describe the temporal development of a cut-in (before, during, and after lane change)?
3. How do established risk indicators differ in timing and severity ranking during cut-ins, and when do they agree or disagree?
4. What is the trade-off between detection quality and computational efficiency for large-scale processing?

---

## The Design: Methods and Procedures

The work follows an empirical, design-oriented methodology:

1. **Data preprocessing:**  
   Load highD tracks, standardize units and coordinate conventions, and link each vehicle to interaction partners using neighbour identifiers.

2. **Lane-change segmentation:**  
   Detect lane-change events from lane assignment over time and derive consistent start/end markers using lateral motion and stabilization rules.

3. **Cut-in scenario identification:**  
   Operationalize a cut-in as a lane change into a target lane where, after completion, the lane-changing vehicle becomes the immediate predecessor of a vehicle in that lane (the new follower). Identify the relevant interaction pair(s) for risk computation (cut-in vehicle with new follower, and optionally with new leader).

4. **Phase-aware scenario windows:**  
   Extract fixed and/or adaptive windows around the manoeuvre. Inspired by phase-based views in the literature [hwang2022fsmrl], scenarios are aligned around lane-boundary crossing and analysed in pre-change, execution, and post-change intervals.

5. **Risk indicator implementation:**  
   Compute time series for TTC, time headway, and braking-demand style indicators (e.g., minimum required deceleration to avoid collision). Add a safe-gap indicator based on minimum safe distance assumptions (reaction distance plus braking distance) as used in lane-merging literature [hwang2022fsmrl].

6. **Evaluation and comparison:**  
   Validate detection on a manually checked subset (quality control) and use it as a small benchmark set for internal comparison across detection/threshold variants. Compare indicators using distribution analysis, correlation, and timing analysis (how early risk escalates). Perform sensitivity checks for thresholds and assumptions.

7. **Framework packaging:**  
   Deliver a documented pipeline with reproducible configuration, scenario exports (event metadata and time series), and visualization utilities for qualitative inspection. Report runtime and scaling behaviour for large-track processing.

---

## Data Collection

This study uses the highD dataset [highd_2018], accessed through the dataset provider's registration/request process for research use. *We were able to complete the access approval already and it has been granted, and the data will be used under the dataset's non-commercial terms.* highD contains 60 drone recordings from highway sections near Cologne, Germany, with trajectories sampled at 25 Hz for more than 110,000 vehicles [highd_2018]. Since highD provides limited upstream/downstream context and does not explicitly include on-ramp/off-ramp geometry, the analysis will treat lane changes primarily as discretionary lane changes within the recorded segment [wanggao2025highd]. No new data will be collected and no human participants are involved.

---

## Data Analysis

The analysis will include:

- descriptive statistics of extracted cut-in features and risk indicators,
- time-aligned comparison of indicator trajectories around manoeuvre completion,
- agreement/disagreement analysis across indicators and severity ranking consistency,
- threshold sensitivity and robustness checks across traffic states and locations,
- computational profiling to report efficiency alongside detection and analysis performance.

---

## Limitations and Delimitations

### Limitations

- highD is limited to short highway segments and does not provide full upstream/downstream context, which restricts conclusions about mandatory vs discretionary lane-change intent [wanggao2025highd].
- Risk indicators rely on modelling assumptions (e.g., TTC under constant-velocity or constant-acceleration) and cannot capture unobserved factors such as turn-signal usage or driver distraction.

### Delimitations

- The study focuses on vehicle–vehicle cut-in interactions on multilane highways.
- The goal is offline scenario detection and analysis rather than an onboard real-time ADAS component.
- Simulation-based re-enactment and OpenSCENARIO export are not included.
- Training new machine learning models is out of scope; the framework focuses on reproducible, rule-based detection and deterministic risk indicator computation.
- Raw highD data will not be redistributed; any shared outputs will be limited to derived annotations and aggregated results, in line with the dataset terms.

---

## Significance of the Study

The thesis contributes a clear and reproducible framework for identifying and analysing cut-in scenarios at scale. Expected contributions include:

- a documented detection pipeline grounded in explicit rules and dataset neighbour relations,
- a curated scenario set extracted from real highway trajectories with computed risk indicators,
- a comparative assessment clarifying how and when common indicators signal risk,
- practical insights supporting safety studies and the design of risk-aware highway driving functions,
- a software engineering oriented implementation that supports repeatable analysis and future extensions.

---

## References

**[highd_2018]**  
Krajewski, R., Bock, J., Kloeker, L., & Eckstein, L. (2018).  
*The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways for Validation of Highly Automated Driving Systems.*  
Proceedings of the IEEE 21st International Conference on Intelligent Transportation Systems (ITSC), 2118–2125.  
https://doi.org/10.1109/ITSC.2018.8569552

**[hayward1972]**  
Hayward, J. C. (1972).  
*Near-miss determination through use of a scale of danger.*  
Highway Research Record.

**[ssam2008]**  
Gettman, D., Pu, L., Sayed, T., & Shelby, S. (2008).  
*Surrogate Safety Assessment Model and Validation.*  
Federal Highway Administration (FHWA), Report FHWA-HRT-08-051.

**[wang2021ssm]**  
Wang, C., Xie, Y., Huang, H., & Liu, P. (2021).  
*A review of surrogate safety measures and their applications in connected and automated vehicles safety modeling.*  
Accident Analysis & Prevention, 157, 106157.  
https://doi.org/10.1016/j.aap.2021.106157

**[johnsson2021validation]**  
Johnsson, C., Laureshyn, A., & D'Agostino, C. (2021).  
*A relative approach to the validation of surrogate measures of safety.*  
Accident Analysis & Prevention, 161, 106350.  
https://doi.org/10.1016/j.aap.2021.106350

**[hwang2022fsmrl]**  
Hwang, S., Lee, K., Jeon, H., & Kum, D. (2022).  
*Autonomous Vehicle Cut-In Algorithm for Lane-Merging Scenarios via Policy-Based Reinforcement Learning Nested Within Finite-State Machine.*  
IEEE Transactions on Intelligent Transportation Systems, 23(10), 17594–17606.  
https://doi.org/10.1109/TITS.2022.3153848

**[shangguan2022]**  
Shangguan, Q., Fu, T., Wang, J., Fang, S., & Fu, L. (2022).  
*A proactive lane-changing risk prediction framework considering driving intention recognition and different lane-changing patterns.*  
Accident Analysis & Prevention, 164, 106500.  
https://doi.org/10.1016/j.aap.2021.106500

**[liu2025multitask]**  
Liu, Y., Zhang, J., Lyu, N., & Zhao, Q. (2025).  
*Predicting lane change maneuver and associated collision risks based on multi-task learning.*  
Accident Analysis & Prevention, 209, 107830.  
https://doi.org/10.1016/j.aap.2024.107830

**[wanggao2025highd]**  
Wang, Z., & Gao, Y. (2025).  
*Lane Change Inconsistencies in the highD Dataset.*  
Findings, March.  
https://doi.org/10.32866/001c.132341  