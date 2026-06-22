# Demo Instructions

## Recommended live demo flow

### 1. Show the Streamlit app

Run:

```powershell
streamlit run app\streamlit_app.py
```

Upload a real image and run:

- YOLO + OpenCV fallback
- OpenCV only

Point out:

- annotated center point
- pick point
- orientation axis
- result table
- runtime
- downloadable output image

### 2. Show the synthetic benchmark

Run:

```powershell
py scripts\run_demo.py --regenerate
```

Open:

```text
outputs/annotated/demo/demo_grid.png
```

Explain that synthetic data gives known ground truth, which makes evaluation measurable.

### 3. Show evaluation metrics

Run:

```powershell
py scripts\evaluate.py
```

Open:

```text
outputs/metrics/evaluation/evaluation_report.md
```

Explain center error, orientation error, pass rates, and inference speed.

### 4. Show robustness testing

Run:

```powershell
py scripts\create_robustness_variants.py --clear
py scripts\evaluate_robustness.py
```

Open:

```text
outputs/metrics/robustness/robustness_report.md
outputs/metrics/robustness/robustness_evaluation_grid.png
```

Explain that occlusion is the dominant failure case.

### 5. Show code quality

Run:

```powershell
py -m pytest
```

Expected:

```text
25 passed
```

## Demo talking points

- The project uses both learned detection and classical geometry.
- Synthetic data provides measurable ground truth.
- Robustness testing shows practical limitations.
- The GUI makes the project easy to demonstrate.
- The code is modular and testable.
