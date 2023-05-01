Here we outline what a typical workflow looks like.

## Creating a client

A Python `velour.Client` object can be created via the following snippet

```python
from velour import Client

client = Client(HOST_URL)
```

In the case that the host uses authentication, then the argument `access_token` should also be passed to `Client`.

## Uploading groundtruth data

To compute metrics, `velour` needs groundtruth labels.
