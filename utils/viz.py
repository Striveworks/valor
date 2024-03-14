import re
from typing import Optional

import matplotlib.ticker as tkr
import pandas as pd
import seaborn as sns


def bytes_to_readable_fmt(x: int, pos: int):
    """
    Convert an integer of bytes into a human-readable format


    Parameters
    ----------
    x
        The number of bytes to convert to human-readable format
    pos
        An unused positional argument that's needed when using this function as a seaborn formatter

    """
    if x < 0:
        return ""
    for x_unit in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024.0:
            return "%3.1f %s" % (x, x_unit)
        x /= 1024.0


def plot_grouped_barchart(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str],
    palette: sns.color_palette = sns.color_palette("tab10"),
    agg_func="mean",
    y_axis_label: Optional[str] = None,
    x_axis_label: Optional[str] = None,
    convert_bytes: bool = False,
    convert_perc: bool = False,
) -> None:
    """
    Plot 2D or 3D data in a barchart

    Parameters
    ----------
    df
        A dataframe containing your profiling data
    x
        The name of the column you want displayed the x-axis
    y
        The name of the column you want displayed the x-axis
    hue
        The name of the column you want to use as a grouper for your legend. Can be None if you only want to graph two dimensions
    palette
        The color palette you want to use for plotting
    agg_func
        The function to use when aggregating your data to the specified number of dimensions
    y_axis_label
        The label you want displayed on the y-axis
    x_axis_label
        The label you want displayed on the x-axis
    convert_bytes
        Converts your y-axis to a human-readable bytes format if True
    convert_perc
        Converts your y-axis to a human-readable percentage format if True
    Returns
    -------
    list
        A list of output records from your profiling function
    """
    df = df.copy(deep=True)

    if not hue:
        df = df.groupby(x, as_index=False)[y].aggregate(agg_func)
        ax = sns.barplot(
            x=x,
            y=y,
            data=df,
            palette=palette,
        )
    else:
        df = df.groupby([x, hue], as_index=False)[y].aggregate(agg_func)
        df[hue] = df[hue].astype(str)
        ax = sns.catplot(
            x=x,
            y=y,
            hue=hue,
            data=df,
            kind="bar",
            palette=palette,
        )

    if not y_axis_label:
        y_axis_label = y

    if not x_axis_label:
        x_axis_label = x

    ax.set(xlabel=x_axis_label, ylabel=y_axis_label)

    if convert_bytes:
        ax.yaxis.set_major_formatter(tkr.FuncFormatter(bytes_to_readable_fmt))
    elif convert_perc:
        ax.yaxis.set_major_formatter(tkr.PercentFormatter(1))


def convert_shortened_bytes_to_int(shortened_bytes_str: str) -> int:
    """Convert a byte-like string (e.g., '64MB') to an int for plotting"""

    unit_mapping = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }

    numeric_part, unit_part = re.match(
        r"([0-9]+)([a-z]+)", shortened_bytes_str, re.I
    ).groups()

    numeric_value = int(numeric_part)

    if unit_part in unit_mapping:
        bytes_value = numeric_value * unit_mapping[unit_part]

        return bytes_value
    else:
        raise ValueError("Invalid unit abbreviation in the input string")