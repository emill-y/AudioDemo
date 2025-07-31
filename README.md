# AudioDemo
# AudioDemo

## Country-by-Voice Demo

A small Gradio app that turns your saved Transformer audio classifier into a playable game:

1. **Place your assets** in the project root:
   * `audio_country_enhanced_transformer_model.keras`
   * `label_encoder_enhanced_transformer.pkl`
   * _(Optional)_ `feature_scaler_transformer.pkl`  – the StandardScaler saved during training.
2. Create & activate a Python 3.10+ virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch locally:

```bash
python app.py
```

4. Deploy for free on [Hugging Face Spaces](https://huggingface.co/spaces):
   * Create a new **Gradio** space.
   * Upload every file in this repo plus the three assets above.
   * Spaces automatically install `requirements.txt` and run `app.py`.
   * After the build completes, share the public Space URL with friends!

Enjoy guessing countries by voice! 🎉
