# Verification report

Date: 2026-10-26. Verification was local on Windows; Linux server deployment and Docker build were not executed.

Environment:

- Python 3.10.0
- PyTorch 2.9.0+cu128; torchvision 0.24.0+cu128
- NVIDIA GeForce RTX 5060 Laptop GPU
- numpy 2.2.6; Pillow 10.3.0; PyYAML 6.0.2; tensorboard 2.20.0

Completed:

1. 16 unittest cases passed, including dynamic fusion gradients, 1-shot graph bounds, support permutation behavior, independent query inference, gradient flow to both classifier branches, reproducible/nonoverlapping sampling, insufficient-data rejection, image/group split leakage, patient-disjoint support/query, checkpoint/config roundtrip, four-task weight objective, stop-gradient consistency, known residual decision, sample-variance CI and validation mode/BatchNorm stability.
2. Full CUDA smoke passed using generated 32px images: Stage1a forward/backward and checkpoint; Stage1b discriminator/feature updates; Stage2 training for 1-shot and 5-shot; evaluation.
3. Frozen encoder parameters AND BatchNorm buffers matched the Stage1b checkpoint exactly after Stage2.
4. Stage2 uninterrupted training and stop/resume produced bitwise-equal head weights in the tested environment.
5. The standalone prediction CLI loaded a Stage2 checkpoint and wrote predictions for five unlabeled query images.
6. Python source compilation succeeded. Shell files use LF without a BOM; the archive is integrity checked.

The smoke accuracy is deliberately not presented as a real-data result. Synthetic images only test numerical and integration behavior. No performance claim on CUB, Malaria, ChestX, CropDisease, EuroSAT, ISIC or miniImageNet has been measured.

Reproduction:

```bash
python -m unittest discover -s tests -v
python -m scripts.smoke --device cuda
```

Not yet verified: Linux driver/runtime compatibility on the user's server, Docker image build, real dataset versions and manifests, published benchmark split equivalence, full training, multi-seed accuracy, baseline comparisons, VRAM/runtime at production image sizes and large shot counts, research novelty and publication suitability.

Exact resume is demonstrated on one software/hardware combination, not guaranteed across PyTorch releases, GPU architectures or operating systems. Checkpoint settings and manifests are checked for consistency. See `requirements-tested.txt` for tested direct dependency versions; it is not a transitive lockfile.
