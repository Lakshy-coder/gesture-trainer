# 🖐️ Hand Gesture Trainer

Train a custom hand-gesture classifier entirely in your browser — record samples from your webcam, train a neural net client-side with TensorFlow.js, and get live predictions. No backend, no build step, no data ever leaves your machine.

[![Live Demo](https://img.shields.io/badge/demo-live-6366f1?style=flat-square)](https://lakshy-coder.github.io/gesture-trainer/)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-in--browser-ff6f00?style=flat-square)](https://www.tensorflow.org/js)
[![No Backend](https://img.shields.io/badge/backend-none-22c55e?style=flat-square)]()

**Live demo:** https://lakshy-coder.github.io/gesture-trainer/

---

## Overview

A from-scratch, framework-free tool for training a binary hand-gesture classifier without leaving the browser tab. Show it your "clone sign" gesture a few times, show it a few "other" poses, hit train, and within seconds it's predicting your gesture live from the webcam feed — all training and inference happen on-device via TensorFlow.js.

It's built to demonstrate a complete applied-ML pipeline end to end: landmark extraction → feature engineering → model training → **proper held-out evaluation** → live inference, entirely client-side.

## How it works

1. **Landmark detection** — [MediaPipe Holistic](https://developers.google.com/mediapipe) tracks 21 3D landmarks per hand from the webcam feed, for both hands at once.
2. **Feature extraction** — each hand's landmarks are made translation- and scale-invariant: every point is re-centered on the wrist (landmark 0) and divided by the wrist-to-middle-knuckle distance (landmark 9). Hand position in frame and distance from the camera stop mattering — only the hand's shape does.
3. **Feature vector** — both normalized hands are concatenated: 21 landmarks × 3 coordinates × 2 hands = **126 features** per frame.
4. **Sample collection** — clicking Record (or pressing `1`/`2`) runs a 3-second countdown, then captures every frame with both hands visible for 4 seconds into one labeled "session."
5. **Training** — a small feed-forward network trains in-browser for 50 epochs (architecture below).
6. **Live inference** — every incoming frame runs through the trained model and shows up as a live confidence bar.

### Model architecture

| # | Layer | Shape | Activation | Params |
|---|---|---|---|---|
| 1 | Dense | 126 → 64 | ReLU | 8,128 |
| – | Dropout (rate 0.3) | 64 → 64 | – | 0 |
| 2 | Dense | 64 → 32 | ReLU | 2,080 |
| 3 | Dense (output) | 32 → 1 | Sigmoid | 33 |

**10,241 trainable parameters total.** Optimizer: Adam · Loss: binary cross-entropy · Epochs: 50 · Batch size: 16.

## Validation methodology

The original version of this project trained on 100% of collected samples and never measured accuracy on anything the model hadn't already seen, so "it works" had no real number behind it. Getting an honest one meant solving a specific problem first:

**Frames within one recording aren't independent.** Each "session" (one countdown + 4-second capture) yields dozens of frames of the *same* pose, in the *same* spot, under the *same* lighting — MediaPipe's landmark smoothing makes consecutive frames nearly identical. Shuffle all frames together and split randomly, and the validation set fills up with near-duplicates of frames the model just trained on. The reported accuracy would be real, but it wouldn't measure what it looks like it measures — generalization to a pose the model hasn't effectively already memorized.

**The fix: session-level stratified splitting.** Every frame is tagged with the ID of the recording session it came from. When building a validation split, whole sessions — never individual frames — get assigned to either train or validation, per class:

```
groupBySession(class)  →  shuffle session IDs  →  hold out ~25% of sessions (min. 1)
                        →  remaining sessions = train, held-out sessions = validation
```

This runs independently per class (stratified), so validation always contains both gestures. A model trains on the train-sessions only, then gets evaluated on the held-out sessions — frames it has never seen anything temporally close to. **Accuracy, precision, recall, F1, and a confusion matrix come from that held-out evaluation** and display in the app before a final model — trained on 100% of the data, for the best live-demo performance — replaces it for on-page testing.

This only activates once you've recorded **3+ separate sessions per class**. With fewer sessions there isn't enough to hold any out safely, so the app still trains a working live model, it just skips reporting a validation number rather than showing a meaningless one.

## Model performance

Open the live demo, record a few sessions per gesture, and click **Train Model** — the validation panel fills in automatically with a confusion matrix alongside these:

| Metric | Value |
|---|---|
| Validation accuracy | _run the app to populate_ |
| Precision | _run the app to populate_ |
| Recall | _run the app to populate_ |
| F1 score | _run the app to populate_ |
| Sessions (train / val) | _run the app to populate_ |

> These numbers depend on your gesture, lighting, and how many sessions you record — that's the point of measuring per-user rather than hardcoding a claim here. Drop your own results into this table once you've trained a model.

## Getting started

### Use the live demo
Open **https://lakshy-coder.github.io/gesture-trainer/** and allow camera access — you can be recording within seconds.

### Run locally
Camera access needs a secure context, so opening `index.html` directly (`file://`) will get blocked by most browsers. Serve it over localhost instead:

```bash
git clone https://github.com/Lakshy-coder/gesture-trainer.git
cd gesture-trainer
npx serve .
# or: python3 -m http.server 8000
```

Then open the printed localhost URL and allow camera access.

## Usage

1. **Collect samples** — click *Record Clone Sign* (or press `1`), get into position during the 3s countdown, hold your gesture through the 4s recording. Repeat 3–5+ times, varying your hand position/angle slightly each time — that variety is what makes validation meaningful. Do the same for *Record Other Poses* (`2`) with a few different non-gesture hand positions.
2. **Train** — click *Train Model*. With 3+ sessions per class recorded, a held-out validation pass runs first and reports metrics; either way, a final model then trains on all your data and goes live.
3. **Test** — hold your gesture in front of the camera and watch the confidence bar respond.
4. **Export/Import** — save recorded samples as JSON to reuse later, or export the trained model (`gesture-model.json` + weights) for use elsewhere.

## Project structure

```
gesture-trainer/
├── index.html   # Page layout, UI elements, CDN script tags (TF.js, MediaPipe)
├── script.js    # Landmark capture, feature extraction, training, validation, inference
├── style.css    # Dark-theme styling
└── README.md
```

Single page, nothing to install, no build step — open it and it runs.

## Tech stack

- **[TensorFlow.js](https://www.tensorflow.org/js)** — in-browser model training & inference
- **[MediaPipe Holistic](https://developers.google.com/mediapipe)** — hand landmark detection
- **Vanilla JavaScript, HTML, CSS** — no framework, no bundler
- **GitHub Pages** — static hosting for the live demo

## Data format

*Export Data* downloads a `gesture-data.json` shaped like:

```json
{
  "formatVersion": 2,
  "exportedAt": "2026-07-10T12:00:00.000Z",
  "samples": {
    "clone_sign": [
      { "x": [ /* 126 floats */ ], "session": "clone_sign_1720000000000" }
    ],
    "not_sign": [
      { "x": [ /* 126 floats */ ], "session": "not_sign_1720000005000" }
    ]
  }
}
```

*Import Data* accepts this format and also still accepts the original v1 export (plain arrays of 126 numbers, no `session` field) for backward compatibility. Imported legacy frames get grouped into one conservative session per class so they can never leak across a validation split — you just won't get true session-level granularity for that particular batch.

## Limitations & future improvements

- **Binary only** — two classes (gesture vs. not). Multiple gestures would need a softmax output instead of sigmoid, plus a UI for arbitrary class labels instead of the fixed "clone sign / other" pair.
- **No rotation invariance** — normalization corrects for hand position and distance from camera, but not for how the hand is rotated/tilted, so orientation still affects the feature vector.
- **Per-frame classification** — each frame is classified independently; there's no temporal model (e.g. a short-window RNN/LSTM), so a single noisy frame can flicker the live prediction.
- **Single hold-out split, not k-fold** — the reported validation number comes from one random session-level split, so it can shift somewhat between training runs, especially with few sessions. More recorded sessions narrow that variance.
- **Session diversity matters** — validation is only as informative as the conditions it's tested under. Sessions all recorded in one sitting, same spot and lighting, will validate more optimistically than real-world use days later would.
- Everything lives in memory client-side; nothing persists across a page reload unless you export it first.

## Credits

Built with [TensorFlow.js](https://www.tensorflow.org/js) and [MediaPipe](https://developers.google.com/mediapipe), both from Google.

## Author

Lakshy Rana — [github.com/Lakshy-coder](https://github.com/Lakshy-coder)

---

⭐ If you found this useful, consider starring the repo.
