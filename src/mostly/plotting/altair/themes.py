from typing import cast

import altair as alt

# inspired by:
# https://github.com/chekos/altair_themes_blog/blob/master/notebooks/urban_theme.py
# https://pmbaumgartner.github.io/streamlitopedia/theming-altair.html

GREYS = {
    "White": "#ffffff",
    "Grey10": "#f8f9fa",
    "Grey20": "#e9ecef",
    "Grey30": "#dee2e6",
    "Grey40": "#ced4da",
    "Grey50": "#adb5bd",
    "Grey60": "#6c757d",
    "Grey70": "#495057",
    "Grey80": "#343a40",
    "Grey90": "#212529",
}


HIGHLIGHTS = {
    "Magma_Magenta": "#b6377a",
}


@alt.theme.register("mostly_light", enable=True)
def mostly_light():
    """Mostly light theme for Altair."""
    font = "Lato"
    font_color = GREYS["Grey90"]
    axis_color = GREYS["Grey90"]
    grid_color = GREYS["Grey30"]
    bg_color = GREYS["White"]
    highlight_color = HIGHLIGHTS["Magma_Magenta"]

    raw = {
        # "width": 450,
        # "height": 400,
        # "autosize": "fit",
        "background": bg_color,
        "config": {
            "title": {
                "anchor": "start",
                "font": font,
                "color": font_color,
                "fontSize": 20,
                "fontStyle": "normal",
                "subtitleFontSize": 16,
                "subtitleColor": font_color,
                "fontWeight": 700,
                "subtitleFontWeight": 400,
                "subtitleFontStyle": "italic",
                "dy": -20,
            },
            "legend": {
                "labelFont": font,
                "labelFontSize": 12,
                "labelColor": font_color,
                "padding": 0,
                "titleFontSize": 12,
                "titleFont": font,
                "titleAnchor": "start",
                "titleFontWeight": 700,
                # "title": "",
            },
            "axis": {
                "titleFontSize": 12,
                "titleFontStyle": "italic",
                "titleFontWeight": 400,
                "titleFont": font,
                "titleColor": font_color,
                "labelFontSize": 12,
                "labelFont": font,
                "labelColor": font_color,
                "labelFontWeight": 400,
                "labelPadding": 5,
            },
            "axisY": {
                "domain": False,
                "grid": True,
                "grid_color": grid_color,
                "gridWidth": 1,
                "ticks": False,
                "titleAngle": 0,
                "titleAlign": "left",
                "titleY": -12,
            },
            "axisX": {
                "domain": True,
                "domainColor": axis_color,
                "domainWidth": 1,
                "grid": False,
                "labelAngle": 0,
                "ticks": True,
                "tickColor": axis_color,
                "tickSize": 3,
            },
            "facet": {"spacing": 50},
            "headerColumn": {
                "title": None,
                "titleFontSize": 14,
                "titleFontWeight": 700,
                "titleFont": font,
                "labelAnchor": "middle",
                "labelFontSize": 12,
                "labelFontWeight": 700,
                "labelFont": font,
            },
            "headerRow": {
                "labelFont": font,
                "labelFontSize": 12,
                "labelAngle": 0,
                "labelAlign": "left",
                "labelAnchor": "middle",
                "labelFontWeight": 700,
                "labelPadding": 25,
                "titleFont": font,
                "titleFontSize": 14,
                "titleFontWeight": 700,
                "titleAngle": 0,
                "orient": "left",
                "title": None,
            },
            "area": {"fill": highlight_color},
            "circle": {"fill": highlight_color},
            "line": {"stroke": highlight_color},
            "rule": {"stroke": GREYS["Grey90"]},
            "path": {"stroke": highlight_color},
            "point": {"stroke": highlight_color},
            "rect": {"fill": highlight_color},
            "shape": {"stroke": highlight_color},
            "symbol": {"fill": highlight_color},
            "bar": {"fill": highlight_color},
            "view": {"stroke": "transparent"},
            "range": {
                "category": {"scheme": "observable10"},  # categorical data 'N'
                "diverging": {"scheme": "spectral"},  # diverging quantitative ramps detected by scale=domainMid
                "heatmap": {"scheme": "magma"},  # quantitative heatmaps "Q"
                "ramp": {"scheme": "cividis"},  # used when not a heatmap
                "ordinal": {"scheme": "lighttealblue"},  # rank-ordered data "O"
            },
        },
    }

    return cast(alt.theme.ThemeConfig, raw)
