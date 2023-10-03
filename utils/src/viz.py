from typing import Optional

import pandas as pd
import seaborn as sns


def plot_grouped_barchart(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str],
    palette: sns.color_palette = sns.color_palette("tab10"),
    agg_func="mean",
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

    Returns
    -------
    list
        A list of output records from your profiling function
    """
    df = df.copy(deep=True)

    if not hue:
        df = df.groupby(x, as_index=False)[y].aggregate(agg_func)
        sns.barplot(x=x, y=y, data=df, palette=palette)
    else:
        df = df.groupby([x, hue], as_index=False)[y].aggregate(agg_func)
        df[hue] = df[hue].astype(str)
        sns.catplot(x=x, y=y, hue=hue, data=df, kind="bar", palette=palette)
