# Evaluation

Code for testing, running inference on validation sets, and metric calculations.

Current component:

- `evaluate_pipeline.py`: compares the target-domain metrics of the source-only classifier and the CycleGAN-augmented classifier.
- `evaluate_multi_pipeline.py`: compares any number of classifier `metrics.json` files, including the target oracle.

Run after both classifier experiments have produced `metrics.json`:

```bash
python src/evaluation/evaluate_pipeline.py \
  --baseline experiments/outputs/classifier_source_only/metrics.json \
  --augmented experiments/outputs/classifier_cyclegan_augmented/metrics.json \
  --out-dir experiments/outputs/evaluation
```

Outputs:

- `experiments/outputs/evaluation/summary.csv`
- `experiments/outputs/evaluation/summary.json`
- `experiments/outputs/evaluation/summary.md`

Figures:

- `figures/metric_scatter_accuracy_f1.png`
- `figures/classifier_temporal_metrics.png`
- `figures/cyclegan_temporal_losses.png`
- `figures/qualitative_translation_previews.png`

Multi-model comparison with the target oracle:

```bash
python src/evaluation/evaluate_multi_pipeline.py \
  --model source_only=experiments/outputs/classifier_source_only/metrics.json \
  --model target_oracle=experiments/outputs/classifier_target_oracle/metrics.json \
  --model cyclegan_augmented=experiments/outputs/classifier_cyclegan_augmented/metrics.json \
  --model vae_gan_augmented=experiments/outputs/classifier_signature_gan_augmented/metrics.json \
  --model adain_gan_augmented=experiments/outputs/classifier_adain_gan_augmented/metrics.json \
  --out-dir experiments/outputs/evaluation_multi \
  --figures-dir figures
```
