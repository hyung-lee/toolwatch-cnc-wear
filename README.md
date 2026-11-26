# TOOLWATCH · CNC Tool Wear Demo

Streamlit image-upload demo trained on real Nonastreda tool photos.

Run: `pip install -r requirements.txt` then `streamlit run app.py`. Python 3.12 recommended.

Training used 256 images from 10 tools. Train tools T1–T6 (148 photos), validation T7–T8 (54), held-out test T9–T10 (54). Four-block CNN regression, 128×128 RGB, AdamW, 60 epochs, validation-selected epoch 18. Test MAE 23.85 µm.

This deployment includes only inference code, trained weights and 54 test photos. No training on the server. See RESULTS.md for actual evaluation.

Remaining observation count is an experimental proxy to the last recorded run, not verified failure RUL. Webcam performance is unvalidated. Replacement review compares predictions against a user-set demonstration threshold.

Dataset: Truchan & Ahmadi, Nonastreda, CC BY 4.0. DOI: https://doi.org/10.17632/m892d2wtzh.1. See DATASET_ATTRIBUTION.md.
