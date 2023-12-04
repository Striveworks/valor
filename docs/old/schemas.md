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

