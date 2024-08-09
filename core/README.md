# valor_core: Compute classification, object detection, and segmentation metrics locally.

Valor is a centralized evaluation store which makes it easy to measure, explore, and rank model performance. Valor empowers data scientists and engineers to evaluate the performance of their machine learning pipelines and use those evaluations to make better modeling decisions in the future.

`valor_core` is the start of a new backbone for Valor's metric calculations. In the future, the Valor API will import `valor_core`'s evaluation functions in order to efficiently compute its classification, object detection, and segmentation metrics. This module offers a few advantages over the existing `valor` evaluation implementations, including:
- The ability to calculate metrics locally, without running separate database and API services
- Faster compute times due to the use of vectors and arrays
- Easier testing, debugging, and benchmarking due to the separation of concerns between evaluation computations and Postgres operations (e.g., filtering, querying)

Valor is maintained by Striveworks, a cutting-edge MLOps company based out of Austin, Texas. We'd love to learn more about your interest in Valor and answer any questions you may have; please don't hesitate to reach out to us on [Slack](https://striveworks-public.slack.com/join/shared_invite/zt-1a0jx768y-2J1fffN~b4fXYM8GecvOhA#/shared-invite/email) or [GitHub](https://github.com/striveworks/valor).

For more information, please see our [user docs](https://striveworks.github.io/valor/).
