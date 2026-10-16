# AI-Based Adaptive Traffic Signal Time Optimization System

A **software-only** college Data Science project. No hardware is used: uploaded
images simulate camera input, uploaded `.wav` files simulate microphone input,
and the traffic signals are simulated in the browser.

> **Limitation:** this is a software prototype. It is NOT a certified traffic
> control system and must not be presented as ready for real roads.

## What is real and what is simulated

| Part | Status |
|---|---|
| Vehicle detection (car / motorcycle / bus / truck) | **Real** - pretrained Ultralytics YOLOv8n runs on your uploaded image |
| Vehicle weights, road capacity, density % | **Simulation assumptions** (`utils/config.py`) |
| Waiting time | **Simulated** - there is no physical sensor; it grows 12 s per analysis cycle (max 60 s) |
| Green-time training labels | **Simulation-based** rule, NOT real-world optimal timings |
| Green-time model | **Real ML** - Random Forest learns from that generated dataset |
| Siren detection | **Real ML** - MFCC features + Random Forest on the audio you upload |
| Siren training data | **Synthetic** audio (see limitation below) |
| Emergency VEHICLE type | **Simulated dropdown** - COCO YOLO has no ambulance / fire-truck class |
| Fixed vs Adaptive comparison, efficiency score | **Simulation only**, not a standard real-world metric |

## Architecture

```
Image -> YOLO -> counts -> density + demand (+ simulated waiting time)
      -> Random Forest -> green time (clamped to 15-60 s) --+
Audio -> MFCC -> siren classifier -> siren ON/OFF ----------+
                                                            v
              emergency vehicle selected AND siren ON ?
                 no  -> Normal: North -> East -> South -> West
                 yes -> Emergency: current -> RED -> safety buffer
                        -> emergency road GREEN -> resume normal sequence
                                                            v
                         Simulation -> Dashboard / What-If
```

**Emergency rule:** Normal Mode follows North -> East -> South -> West.
Emergency Mode temporarily overrides that sequence only when an emergency
vehicle is selected AND its siren is classified as ON. After the emergency
event the controller safely returns to the normal sequence. A visible
ambulance with no siren never triggers emergency mode.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000/ . On the first analysis, `yolov8n.pt` (~6 MB)
downloads automatically, so you need internet once.

The trained models and datasets are already included. To rebuild them:

```bash
python -m ml.dataset_generator        # simulation-based traffic dataset
python -m ml.train_traffic_model      # Linear Regression vs Random Forest + plots
python -m ml.generate_siren_dataset   # synthetic siren / no_siren .wav clips
python -m ml.train_siren_model        # MFCC + Random Forest siren classifier
```

If a saved model cannot be loaded (for example it was saved with a different
scikit-learn version), the app automatically retrains it from the bundled
datasets the first time it is needed (a few seconds), so this is not an error.

## Using the app

1. **Upload** - one image per road. Optionally pick an emergency vehicle and
   upload a `.wav` per road. Sample clips: `data/audio/siren/` and
   `data/audio/no_siren/`.
2. **Analysis** - boxes, counts, density, demand, predicted green time, emergency status.
3. **Simulation** - START / STOP / RESTART the four-signal countdown.
4. **Dashboard** - model metrics, plots, EDA charts, Fixed vs Adaptive, emergency log.
5. **What-If** - change a road's traffic and see the new predicted green time.
6. To simulate another cycle, go back to Upload and analyze new images (waiting time grows each cycle).

### Emergency test cases

| Emergency vehicle | Siren audio | Result |
|---|---|---|
| Ambulance | siren clip | **Emergency priority ACTIVE** |
| Ambulance | no_siren clip / none | Normal mode |
| Fire Truck | siren clip | **Emergency priority ACTIVE** |
| Fire Truck | no_siren clip / none | Normal mode |
| None | siren clip | Normal mode |

## Project structure

```
app.py                     Flask routes only (thin orchestration)
detection/                 YOLO detector + OpenCV box drawing
audio/                     audio loading, MFCC features, siren detector
ml/                        dataset generator, training scripts, predictor
traffic/                   density, demand, timing, signal + emergency controller, comparison
utils/                     config.py (all constants), upload validators
templates/ static/         pages, CSS, simulation.js, dashboard.js (Plotly via CDN)
models/                    traffic_model.pkl, siren_model.pkl
data/traffic/              training_data.csv, model_metrics.json, plots/
data/audio/                siren/ and no_siren/ clips
```

No database is used. The latest analysis and the emergency log live in memory
and reset when the app restarts.

## How the green-time label is generated (honest explanation)

`ml/dataset_generator.py` creates 1,200 random traffic situations, then
computes

```
demand = car*1 + motorcycle*0.5 + bus*2 + truck*2 + waiting_time*0.5
green  = 15 + (demand / max_demand) * (60 - 15)   (+ small noise, clipped to 15-60)
```

The models therefore learn to approximate this documented rule. Because the rule
is linear, the Linear Regression baseline scores slightly higher than the Random
Forest on the held-out test set (R^2 about 0.84 vs 0.81 in the included run).
Random Forest is still the deployed model because the project specifies it as
primary, and the numbers are reported unmodified. A good R^2 shows the model
learned the rule; it does NOT show the timings are optimal for real traffic.

## Known limitations

- **Siren classifier** is trained on small *synthetic* audio (frequency-sweeping
  tones vs steady tones/noise). Its ~100% test accuracy reflects how easy that
  synthetic task is. It is not validated on real ambulance or fire-truck
  recordings and will not identify every real-world siren.
- **YOLOv8n** is small and fast, so dense or occluded scenes are under-counted.
- **Random Forest** is not strictly monotonic: in some low-traffic regions,
  adding vehicles can slightly lower the predicted time (a limitation worth
  mentioning in a viva). Predictions are always clamped to 15-60 s.
- The safety transition is a simplified software model, not a real controller.
