# Copyright (c) 2023 Christopher Prohm
# From blogpost: https://cprohm.de/blog/polars-seaborn/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import functools as ft
from dataclasses import dataclass
from typing import Union

import polars as pl
import seaborn as sb


@pl.api.register_dataframe_namespace("sb")
@pl.api.register_lazyframe_namespace("sb")
@dataclass
class SeabornPlotting:
    df: Union[pl.DataFrame, pl.LazyFrame]

    def pipe(self, func, /, **kwargs):
        def maybe_collect(df):
            return df.collect() if isinstance(df, pl.LazyFrame) else df

        exprs = {}
        for key in "x", "y", "hue", "col", "row":
            val = kwargs.get(key)
            if val is None:
                continue

            expr = pl.col(val) if isinstance(val, str) else val

            exprs[expr.meta.output_name()] = expr
            kwargs[key] = expr.meta.output_name()

        return self.df.select(list(exprs.values())).pipe(maybe_collect).to_pandas().pipe(func, **kwargs)

    relplot = ft.partialmethod(pipe, sb.relplot)
    scatterplot = ft.partialmethod(pipe, sb.scatterplot)
    lineplot = ft.partialmethod(pipe, sb.lineplot)
    displot = ft.partialmethod(pipe, sb.displot)
    histplot = ft.partialmethod(pipe, sb.histplot)
    kdeplot = ft.partialmethod(pipe, sb.kdeplot)
    ecdfplot = ft.partialmethod(pipe, sb.ecdfplot)
    rugplot = ft.partialmethod(pipe, sb.rugplot)
    distplot = ft.partialmethod(pipe, sb.distplot)
    catplot = ft.partialmethod(pipe, sb.catplot)
    stripplot = ft.partialmethod(pipe, sb.stripplot)
    swarmplot = ft.partialmethod(pipe, sb.swarmplot)
    boxplot = ft.partialmethod(pipe, sb.boxplot)
    violinplot = ft.partialmethod(pipe, sb.violinplot)
    boxenplot = ft.partialmethod(pipe, sb.boxenplot)
    pointplot = ft.partialmethod(pipe, sb.pointplot)
    barplot = ft.partialmethod(pipe, sb.barplot)
    countplot = ft.partialmethod(pipe, sb.countplot)
    lmplot = ft.partialmethod(pipe, sb.lmplot)
    regplot = ft.partialmethod(pipe, sb.regplot)
    residplot = ft.partialmethod(pipe, sb.residplot)
