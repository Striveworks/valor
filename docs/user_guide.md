# API

The backend consists of three components

1. A `velour` REST API service.
2. A PostgreSQL instance with the PostGIS extention.
3. A redis instance

FastAPI generates API documentation which can be found at:

`http://<url>:<port>/docs#/`


# Authentication

The API can be run without authentication (by default) or with authentication provided by [auth0](https://auth0.com/). A small react app (code at `web/`)

## Backend

To enable authentication for the backend either set the environment variables `AUTH_DOMAIN`, `AUTH_AUDIENCE`, and `AUTH_ALGORITHMS` or put them in a file named `.env.auth` in the `api` directory. An example of such a file is

```
AUTH0_DOMAIN="velour.us.auth0.com"
AUTH0_AUDIENCE="https://velour.striveworks.us/"
AUTH0_ALGORITHMS="RS256"
```


## Testing auth

All tests mentioned above run without authentication except for `integration_tests/test_client_auth.py`. Running this test requires setting the envionment variables `AUTH0_DOMAIN`, `AUTH0_AUDIENCE`, `AUTH0_CLIENT_ID`, and `AUTH0_CLIENT_SECRET` accordingly.

# Deployment settings

For deploying behind a proxy or with external routing, the environment variable `API_ROOT_PATH` can be set in the backend, which sets the `root_path` arguement to `fastapi.FastAPI` (see https://fastapi.tiangolo.com/advanced/behind-a-proxy/#setting-the-root_path-in-the-fastapi-app)

# Schemas

# Geometry

<details>
<summary><strong>Point</strong></summary>

## Description

Briefly describe the purpose and functionality of the class.

## Attributes

>| name | type | description |
>| - | - | - |
>| x | `float` |  |
>| y | `float` |  |

## Methods

><details>
><summary><b>resize</b></summary>
>
>**Description**\
>Initialize the class instance.
>
>**Parameters**
>| name | type | description |
>| - | - | - |
>| og_img_h | `int` |  |
>| og_img_w | `int` |  |
>| new_img_h | `int` |  |
>| new_img_w | `int` |  |
>
>**Returns**\
>None.
></details>

## Usage

```python
# Creating an instance of MyClass
my_instance = MyClass(param1=value1, param2=value2)
```

</details>

<details>
<summary><strong>Box</strong></summary>

## Description

Briefly describe the purpose and functionality of the class.

## Attributes

>| name | type | description |
>| - | - | - |
>| min | `Point` |  |
>| max | `Point` |  |
>

## Usage

```python
# Creating an instance of MyClass
my_instance = MyClass(param1=value1, param2=value2)
```
</details>

<details>
<summary><strong>BasicPolygon</strong></summary>

## Description

Briefly describe the purpose and functionality of the class.

## Attributes

>| name | type | description |
>| - | - | - |
>| points | `List[Point]` |  |

## Methods

><details>
><summary><b>xy_list</b></summary>
>
>**Description**\
>Initialize the class instance.
>
>**Returns**\
>`List[Point]`
></details>

><details>
><summary><b>tuple_list</b></summary>
>
>**Description:**\
>Initialize the class instance.
>
>**Returns:**\
>`List[Tuple[float,float]]`
></details>

## Usage

```python
# Creating an instance of MyClass
my_instance = MyClass(param1=value1, param2=value2)
```

</details>

<details>
<summary><strong>Polygon</strong></summary>
</details>

<details>
<summary><strong>BoundingBox</strong></summary>
</details>

<details>
<summary><strong>MultiPolygon</strong></summary>
</details>

<details>
<summary><strong>Raster</strong></summary>
</details>

