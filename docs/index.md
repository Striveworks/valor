# Introduction

Valor is a collection of evaluation methods that make it easy to measure, explore, and rank machine learning model performance. Valor empowers data scientists and engineers to evaluate the performance of their machine learning pipelines and use those evaluations to make better modeling decisions in the future. To skip this textual introduction and dive right in, first go [here](#installation) for basic installation instructions, and then checkout the [example notebooks](https://github.com/Striveworks/valor/blob/main/examples/).

Valor is maintained by Striveworks, a cutting-edge machine learning operations (MLOps) company based out of Austin, Texas. We'd love to learn more about your interest in Valor and answer any questions you may have; please don't hesitate to reach out to us on [Slack](https://striveworks-public.slack.com/join/shared_invite/zt-1a0jx768y-2J1fffN~b4fXYM8GecvOhA#/shared-invite/email) or [GitHub](https://github.com/striveworks/valor).

## Installation

### PyPi
```
pip install valor-lite
```

### Source
```
git clone https://github.com/Striveworks/valor.git
cd valor
make install
```

## Quick Links

- **Documentation**
    - Classification
        - [DataLoader](classification/dataloader.md)
        - [Metrics](classification/metrics.md)
    - Object Detection
        - [Metrics](object_detection/metrics.md)
    - Semantic Segmentation
        - [Metrics](semantic_segmentation/metrics.md)
    - Text Generation
        - [Metrics](text_generation/metrics.md)
- **[Example Notebooks](https://github.com/Striveworks/valor/blob/main/examples/)**: Collection of descriptive Jupyter notebooks giving examples of how to evaluate model performance using Valor.
- **[Contributing and Development](contributing.md)**: Explains how you can build on and contribute to Valor.
