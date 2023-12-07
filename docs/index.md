# Introduction

Velour is a centralized evaluation store which makes it easy to measure, explore, and rank model performance. Velour empowers data scientists and engineers to evaluate the performance of their machine learning pipelines and use those evaluations to make better modeling decisions in the future.

These docs are organized as follows:

- **[Overview](index.md)** (this page): Provides an overview of what Velour is, why it's important, and how it works.
- **[Getting Started](getting_started.md)**: Details everything you need to get up-and-running with Velour.
- **[Technical Concepts](technical_conepts.md)**: Describes on the technical details that underpin Velour.
- **[Contributing & Development](getting_started.md)**: Explains how you can build on and contribute to Velour.
- **[References](references.md)**: Shares reference documentation for our API and Python client.

# Overivew of Velour

In this section, we'll explore what Velour is, why it's important, and how it works. This overview is available in a quick 5-minute video below:

<video controls>
  <source src="static/Velour_Video_Demo.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>

![A video overview of Velour](static/Velour_Video_Demo.mp4)

# Use Cases for an Evaluation Store

As we worked with dozens of data scientists and engineers on their MLOps pipelines, we identified three important questions that an effective evaluation store could help them answer. First, they wanted to understand "Of the various models I tested for a given dataset, which one performs best?". This is a very common and important use case, and one that is often solved in a local Jupyter notebook. Jupyter works well for this use case up until a user inevitably wants to compare a new model evaluation to a very old evaluation, in which case they'll often experience traceability issues that lead to apples-to-oranges comparisons.

Second, our users wanted to understand "how does
Second: how does performance vary across datasets? In a notebook-centric world, it’s difficult to identify patterns across datasets and use cases because you don’t have an easy way to run apples-to-apples comparisons across disparate pipelines
Finally, we get to the most important question: given a completely new use case, how might we use prior evaluations to pick the best model for our new ML pipeline? This third question requires the ability to filter evaluations on very granular criteria (like time of day, geospatial coordinates) to pick the best-suited model and accelerate development

# Why Velour

# How It Works

# Illustrative example

 We’re starting with computer vision, but will support additional evaluations in the future.




## How it works
The core of _velour_ is a backend REST API service. user's will typically interact with this service via a python client. there is also a lightweight web interface. At a high-level, the typical workflow involves posting groundtruth annotations (class labels, bounding boxes, segmentation masks, etc.) and model predictions to the service. Velour, on the backend, then handles the computation of metrics, stores them centrally, and allows them to be queried. Velour does _not_ store raw data (such as underlying images) or facilitate model inference. It only stores groundtruth annotations and the predictions outputted from a model.

Some highlights:

- The service handles the computation of metrics. This help makes them trustworthy and auditable, and is also useful when metric computations can be computationally expensive (e.g. for object detection).
- Metrics are centralized and queryable. In particular, the service facilicates comparing performance of multiple models against multiple datasets.
- Since inferences and groundtruths are stored, additional metrics can be computed without having to redo model inferences. For example, maybe you run default AP metric settings for object detection but later decide you want to know AP at lower IOU thresholds.
