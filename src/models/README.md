# Models

Neural network architectures (e.g., definitions of `nn.Module` classes) and model components.

Current components:

- `classifier.py`: `SmallMRIClassifier`, the CNN used for CN vs AD classification from center MRI slices.
- `cyclegan.py`: `Generator` and `Discriminator` modules used for scanner-style image translation.

Trained checkpoint files are not stored here. They are written to `experiments/checkpoints/`.
