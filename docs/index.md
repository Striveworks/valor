# Introduction

Velour is a centralized evaluation store which makes it easy to measure, explore, and rank model performance. Velour empowers data scientists and engineers to evaluate the performance of their machine learning pipelines and use those evaluations to make better modeling decisions in the future. [Click here](https://github.com/Striveworks/velour/blob/main/examples/getting_started.ipynb) to skip this textual introduction and dive right into our sample notebook.

Velour is maintained by Striveworks, a cutting-edge MLOps company based out of Austin, Texas. We'd love to learn more about your interest in Velour and answer any questions you may have; please don't hesitate to reach out to us on [Slack](https://striveworks-public.slack.com/join/shared_invite/zt-1a0jx768y-2J1fffN~b4fXYM8GecvOhA#/shared-invite/email) or [GitHub](https://github.com/striveworks/velour).


These docs are organized as follows:

- **[Overview](index.md)** (this page): Provides an overview of what Velour is, why it's important, and how it works.
- **[Sample Notebook](https://github.com/Striveworks/velour/blob/main/examples/getting_started.ipynb)**: Illustrates how to evaluate model performance using Velour in a descriptive Jupyter notebook.
- **[Getting Started](getting_started.md)**: Details everything you need to get up-and-running with Velour.
- **[Metrics](https://striveworks.github.io/velour/metrics)**: Describes all of the metrics that you can calculate using Velour.
- **[Endpoints](endpoints.md)**: Documents Velour's various API endpoints.
- **[Technical Concepts](technical_concepts.md)**: Describes the technical concepts that underpin Velour.
- **[Contributing & Development](contributing.md)**: Explains how you can build on and contribute to Velour.
- **[Python Client API](client_api/Client.md)**: Shares reference documentation for our Python client.

# Overview

In this section, we'll explore what Velour is, why it's important, and provide a high-level description of how it works. This overview is also available in the following five-minute video:

<div style="position: relative; width: 100%; height: 0; padding-bottom: 56%;">
  <iframe style="position: absolute; width: 100%; height: 100%; left: 0; top: 0;" src="https://www.youtube.com/embed/SbJtZ9osFkg?si=P4SLqdspuTJ0sXbb" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>

# Use Cases for a Containerized Evaluation Store

As we've worked with dozens of data scientists and engineers on their MLOps pipelines, we identified three important questions that an effective evaluation store could help them answer. First, they wanted to understand **"Of the various models I tested for a given dataset, which one performs best?"**. This is a very common and important use case, and one that is often solved on a model-to-model basis in a local Jupyter notebook. This focus on bespoke implementations limits traceability and makes it difficult to create apples-to-apples comparisons between new model runs and prior model runs.

Second, our users wanted to understand **"How does the performance of a particular model vary across datasets?"**. We found that many practitioners use the same computer vision model (e.g., YOLOv8) for a variety of supervised learning tasks, and they needed a way to identify patterns where that particular model didn't meet expectations.

Finally, our users wanted to understand **"How can I use my prior evaluations to pick the best model for a future ML pipeline?"**. This last question requires the ability to filter previous evaluations on granular metadata (e.g., time of day, geospatial coordinates, etc.) in order to provide tailored recommendations regarding which model to pick in the future.

With these three use cases in mind, we set out to build a centralized evaluation store that we later named Velour.

# Introducing Velour

Velour is a centralized evaluation store which makes it easy to measure, explore, and rank model performance. Our ultimate goal with Velour is to help data scientists and engineers pick the right ML model for their specific needs. To that end, we built Velour with three design principles in mind:

- **Velour works with any dataset or model**: We believe Velour should be able to handle any supervised learning task that you want to throw at it. Just pass in your groundtruth annotations and predictions, describe your learning task (i.e., object detection), and Velour will do the rest. (Note: at launch, Velour will only support classification and computer vision (i.e., image segmentation and object detection) tasks. We're confident this framework will abstract well to other supervised learning tasks, and plan to support them in later releases).
- **Velour can handle any type of image, model, or dataset metadata you throw at it**: Metadata is a critical component of any evaluation store as it enables the system to offer tailored model recommendations based on a user's specific needs. To that end, we built Velour to handle any metadata under the sun. Dates, geospatial coordinates, and even JSONs filled with config details are all on the table. This means you can slice-and-dice your evaluations any way you want: just pass in the right labels for your use case, define your filter (say a geographic bounding box) and you’ll get back results for your specific needs.
- **Velour standardizes the evaluation process**: The trickiest part of comparing two different model runs is avoiding apples-to-oranges comparisons. Velour helps you audit your metrics and avoid false comparisons by versioning your uploads, storing them in a centralized location, and ensuring that you only compare runs that used the exact same filters and metrics.


# How It Works: An Illustrative Example

Let’s walk through a quick example to bring Velour to life.

Say that you're interested in using computer vision models to detect forest fires around the world using satellite imagery. You've just been tasked with building a new ML pipeline to detect fires in an unfamiliar region of interest. How might you leverage your evaluation metrics from prior ML pipelines to understand which model will perform best for this particular use case?


<img src="static/example_1.png" alt="A satellite image of forest fires.">

To answer this question, we'll start by passing-in three pieces of information from each of our prior modeling runs:

- **GroundTruths**: First, we'll pass-in human-annotated bounding boxes to tell Velour exactly where forest fires can be found across all of the satellite images used in prior runs.
- **Predictions**: Next, we'll pass machine-generated predictions for each image (also in the form of bounding boxes) so that Velour can evaluate how well each model did at predicting forest fires.
- **Labels**: Finally, we'll pass metadata to Velour describing each of our various images (e.g., the time of day the photo was taken, the geospatial coordinates of the forest in the photo, etc.). We'll use this metadata later on in order to identify the right model for our new use case.

Once we pass-in these three ingredients, Velour will compare all of our `GroundTruths` and `Predictions` in order to calculate various evaluation metrics (i.e., mean average precision, or mAP). These metrics, `Labels`, `GroundTruths`, and `Predictions` will all be stored in Postgres, with PostGIS support for fast geospatial lookups and geometric comparisons at a later date.

Finally, once all of our previous pipeline runs and evaluations are stored in Velour, we can use Velour’s API to specify our exact filter criteria and get back its model rankings. In this case, we can ask Velour to find us the best model for detecting forest fires at night in a 50 mile radius around (42.36, -71.03), sorted by mAP. Velour will then filter all of our stored evaluation metrics, rank each model with evaluations that meet our criteria, and send back all relevant evaluation metrics to help us determine which model to use for our new modeling pipeline.


<img src="static/example_2.png" alt="A satellite image of forest fires.">

# Next Steps

To get started with Velour, we'd recommend reviewing our [sample notebooks](https://github.com/Striveworks/velour/blob/main/examples/getting_started.ipynb) or reading our [Getting Started docs](getting_started.md). For more detailed explainations of Velour's technical underpinnings, see our [technical concepts guide](technical_concepts.md).
