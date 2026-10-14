from pathlib import Path
import json
import re
import time

import numpy as np
import pandas as pd

from django.core.paginator import Paginator
from django.shortcuts import render


# ============================================================
# PATHS / SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "dataset" / "cumulative.csv"

PAGE_SIZE = 25
MAX_SCATTER_POINTS = 3000


# ============================================================
# COLUMNS
# ============================================================

DISPLAY_COLUMNS = [
    "rowid",
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_score",
    "koi_period",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_steff",
    "koi_srad",
    "ra",
    "dec",
    "koi_kepmag",
]

NUMERIC_COLUMNS = [
    "rowid",
    "kepid",
    "koi_score",
    "koi_period",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_steff",
    "koi_srad",
    "ra",
    "dec",
    "koi_kepmag",
]

VISUALIZATION_COLUMNS = [
    "koi_disposition",
    "koi_prad",
    "koi_period",
    "koi_teq",
    "koi_steff",
    "koi_srad",
]


# ============================================================
# GLOBAL DATA
# ============================================================

EXOPLANET_DF = pd.DataFrame()

TOTAL_OBJECTS = 0
CONFIRMED_COUNT = 0
CANDIDATE_COUNT = 0
FALSE_POSITIVE_COUNT = 0

DATASET_ERROR = None

SEARCH_COLUMNS = []

VISUALIZATION_DATA = {
    "disposition_labels": [],
    "disposition_values": [],
    "radius_data": [],
    "period_data": [],
    "temperature_data": [],
    "scatter_period": [],
    "scatter_radius": [],
    "stellar_temperature": [],
    "stellar_radius": [],
}

# Prepared once when the dataset is loaded.
COMPARISON_DATA = {}

COMPARISON_PLANETS = []


# ============================================================
# GENERAL HELPERS
# ============================================================

def _json_dumps(value):
    """Return JSON-safe output."""
    return json.dumps(value, allow_nan=False)


def clean_json_data(values):
    """
    Convert numeric pandas Series/list to JSON-safe floats.
    """
    if values is None:
        return []

    series = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    )

    series = series.dropna()
    series = series[np.isfinite(series)]

    return [
        float(value)
        for value in series.tolist()
    ]


def _prepare_numeric_series(df, column):
    """
    Return a positive, finite numeric Series.
    """
    if column not in df.columns:
        return pd.Series(dtype="float64")

    series = pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna()

    series = series[np.isfinite(series)]

    return series[series > 0]


def _sample_series(
    series,
    maximum=MAX_SCATTER_POINTS,
):
    """
    Downsample large datasets for visualization.
    """
    if len(series) <= maximum:
        return series

    return series.sample(
        n=maximum,
        random_state=42,
    )


# ============================================================
# COMPARISON HELPERS
# ============================================================

def _comparison_number(value, decimals=3):
    """
    Convert pandas/numpy values into JSON-safe numbers.

    Missing/invalid values become None.
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None

    try:
        number = float(value)

        if not np.isfinite(number):
            return None

        return round(number, decimals)

    except (TypeError, ValueError):
        return None


def _comparison_text(value):
    """
    Convert dataset text into clean display text.
    """
    if value is None:
        return "Not available"

    try:
        if pd.isna(value):
            return "Not available"
    except (TypeError, ValueError):
        return "Not available"

    text = str(value).strip()

    return text if text else "Not available"


def _build_comparison_record(row):
    """
    Convert one dataset row into the structure used
    by comparison.html.
    """

    kepler_name = _comparison_text(
        row.get("kepler_name")
    )

    kepoi_name = _comparison_text(
        row.get("kepoi_name")
    )

    # Prefer official Kepler name.
    if kepler_name != "Not available":
        display_name = kepler_name

    # Otherwise use KOI identifier.
    elif kepoi_name != "Not available":
        display_name = kepoi_name

    else:
        display_name = "Not available"

    return {
        "name": display_name,

        "kepler_name": kepler_name,

        "kepoi_name": kepoi_name,

        "kepid": _comparison_number(
            row.get("kepid"),
            0,
        ),

        "koi_disposition": _comparison_text(
            row.get("koi_disposition")
        ),

        "koi_score": _comparison_number(
            row.get("koi_score"),
            3,
        ),

        "koi_prad": _comparison_number(
            row.get("koi_prad"),
            3,
        ),

        "koi_teq": _comparison_number(
            row.get("koi_teq"),
            2,
        ),

        "koi_insol": _comparison_number(
            row.get("koi_insol"),
            3,
        ),

        "koi_period": _comparison_number(
            row.get("koi_period"),
            3,
        ),

        "koi_steff": _comparison_number(
            row.get("koi_steff"),
            2,
        ),

        "koi_srad": _comparison_number(
            row.get("koi_srad"),
            3,
        ),
    }


def prepare_comparison_data(df):
    """
    Prepare comparison data once.

    This is much more efficient than rebuilding the
    comparison dictionary every time the comparison
    page is opened.
    """

    comparison_data = {}

    if df.empty:
        return comparison_data

    required_columns = [
        "kepler_name",
        "kepoi_name",
        "koi_disposition",
        "koi_score",
        "koi_prad",
        "koi_teq",
        "koi_insol",
        "koi_period",
        "koi_steff",
        "koi_srad",
        "kepid",
    ]

    available_columns = [
        column
        for column in required_columns
        if column in df.columns
    ]

    if not available_columns:
        return comparison_data

    # itertuples() is faster than iterrows().
    for row in df[available_columns].itertuples(
        index=False,
        name="ExoplanetRow",
    ):

        row_data = row._asdict()

        kepler_name = _comparison_text(
            row_data.get("kepler_name")
        )

        kepoi_name = _comparison_text(
            row_data.get("kepoi_name")
        )

        if kepler_name != "Not available":
            key = kepler_name

        elif kepoi_name != "Not available":
            key = kepoi_name

        else:
            continue

        # Keep the first record for a unique planet name.
        if key not in comparison_data:
            comparison_data[key] = _build_comparison_record(
                row_data
            )

    return comparison_data


def prepare_comparison_planets(comparison_data):
    """
    Prepare lightweight selector data.
    """

    planets = []

    for name, data in comparison_data.items():

        planets.append({
            "name": name,

            "kepler_name": data.get(
                "kepler_name",
                "Not available",
            ),

            "kepoi_name": data.get(
                "kepoi_name",
                "Not available",
            ),

            "koi_disposition": data.get(
                "koi_disposition",
                "Not available",
            ),
        })

    planets.sort(
        key=lambda item: item["name"].lower()
    )

    return planets


# ============================================================
# DIFFERENCE HELPERS
# ============================================================

def _comparison_difference(value1, value2):
    """
    Calculate absolute and percentage difference.

    Percentage uses the larger absolute value as
    the reference.
    """

    if value1 is None or value2 is None:
        return {
            "absolute": None,
            "percentage": None,
        }

    try:
        value1 = float(value1)
        value2 = float(value2)

    except (TypeError, ValueError):
        return {
            "absolute": None,
            "percentage": None,
        }

    if not np.isfinite(value1) or not np.isfinite(value2):
        return {
            "absolute": None,
            "percentage": None,
        }

    absolute = abs(value1 - value2)

    denominator = max(
        abs(value1),
        abs(value2),
    )

    if denominator == 0:
        percentage = 0.0
    else:
        percentage = (
            absolute / denominator
        ) * 100

    return {
        "absolute": round(absolute, 3),
        "percentage": round(percentage, 2),
    }


def build_selected_comparison(
    planet1,
    planet2,
):
    """
    Build dynamic comparison information
    for two selected planets.
    """

    metrics = {
        "radius": (
            planet1.get("koi_prad"),
            planet2.get("koi_prad"),
        ),

        "temperature": (
            planet1.get("koi_teq"),
            planet2.get("koi_teq"),
        ),

        "period": (
            planet1.get("koi_period"),
            planet2.get("koi_period"),
        ),

        "insolation": (
            planet1.get("koi_insol"),
            planet2.get("koi_insol"),
        ),
    }

    differences = {}

    for metric, values in metrics.items():

        differences[metric] = _comparison_difference(
            values[0],
            values[1],
        )

    return {
        "planet1": planet1,
        "planet2": planet2,
        "differences": differences,
    }


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset():

    global EXOPLANET_DF
    global TOTAL_OBJECTS
    global CONFIRMED_COUNT
    global CANDIDATE_COUNT
    global FALSE_POSITIVE_COUNT
    global DATASET_ERROR
    global SEARCH_COLUMNS
    global VISUALIZATION_DATA
    global COMPARISON_DATA
    global COMPARISON_PLANETS

    started = time.perf_counter()

    try:

        print(
            "\n" + "=" * 70,
            flush=True,
        )

        print(
            "START DATASET LOAD",
            flush=True,
        )

        print(
            f"CSV PATH: {CSV_PATH}",
            flush=True,
        )

        # ----------------------------------------------------
        # Check CSV
        # ----------------------------------------------------

        if not CSV_PATH.is_file():

            raise FileNotFoundError(
                f"Dataset file not found: {CSV_PATH}"
            )

        # ----------------------------------------------------
        # Read CSV once
        # ----------------------------------------------------

        raw_df = pd.read_csv(
            CSV_PATH,
            low_memory=False,
        )

        print(
            f"DATASET SHAPE: {raw_df.shape}",
            flush=True,
        )

        print(
            f"TOTAL CSV COLUMNS: {len(raw_df.columns)}",
            flush=True,
        )

        raw_df.columns = (
            raw_df.columns
            .astype(str)
            .str.strip()
        )

        # ----------------------------------------------------
        # Select required columns
        # ----------------------------------------------------

        required_columns = [
            column
            for column in dict.fromkeys(
                DISPLAY_COLUMNS + VISUALIZATION_COLUMNS
            )
            if column in raw_df.columns
        ]

        if not required_columns:

            raise ValueError(
                "None of the expected exoplanet columns "
                "were found in cumulative.csv."
            )

        df = raw_df[
            required_columns
        ].copy()

        # ----------------------------------------------------
        # Clean text
        # ----------------------------------------------------

        text_columns = [
            "kepoi_name",
            "kepler_name",
            "koi_disposition",
        ]

        for column in text_columns:

            if column in df.columns:

                df[column] = (
                    df[column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

        if "koi_disposition" in df.columns:

            df["koi_disposition"] = (
                df["koi_disposition"]
                .str.upper()
            )

        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------

        for column in NUMERIC_COLUMNS:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

        # ----------------------------------------------------
        # Search text
        # ----------------------------------------------------

        search_columns = [
            column
            for column in DISPLAY_COLUMNS
            if column in df.columns
        ]

        if search_columns:

            df["_search_text"] = (
                df[search_columns]
                .fillna("")
                .astype(str)
                .agg(
                    " ".join,
                    axis=1,
                )
                .str.lower()
            )

        else:

            df["_search_text"] = ""

        SEARCH_COLUMNS = search_columns

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        TOTAL_OBJECTS = int(len(df))

        if "koi_disposition" in df.columns:

            counts = (
                df["koi_disposition"]
                .value_counts()
            )

            CONFIRMED_COUNT = int(
                counts.get(
                    "CONFIRMED",
                    0,
                )
            )

            CANDIDATE_COUNT = int(
                counts.get(
                    "CANDIDATE",
                    0,
                )
            )

            FALSE_POSITIVE_COUNT = int(
                counts.get(
                    "FALSE POSITIVE",
                    0,
                )
            )

        else:

            CONFIRMED_COUNT = 0
            CANDIDATE_COUNT = 0
            FALSE_POSITIVE_COUNT = 0

        # ----------------------------------------------------
        # Store main dataframe
        # ----------------------------------------------------

        EXOPLANET_DF = df

        # ----------------------------------------------------
        # Prepare visualization data ONCE
        # ----------------------------------------------------

        VISUALIZATION_DATA = (
            prepare_visualization_data(df)
        )

        # ----------------------------------------------------
        # Prepare comparison data ONCE
        # ----------------------------------------------------

        COMPARISON_DATA = (
            prepare_comparison_data(df)
        )

        COMPARISON_PLANETS = (
            prepare_comparison_planets(
                COMPARISON_DATA
            )
        )

        DATASET_ERROR = None

        elapsed = (
            time.perf_counter()
            - started
        )

        print(
            f"TOTAL RECORDS: {TOTAL_OBJECTS}",
            flush=True,
        )

        print(
            f"CONFIRMED: {CONFIRMED_COUNT}",
            flush=True,
        )

        print(
            f"CANDIDATES: {CANDIDATE_COUNT}",
            flush=True,
        )

        print(
            f"FALSE POSITIVE: {FALSE_POSITIVE_COUNT}",
            flush=True,
        )

        print(
            f"COMPARISON PLANETS: "
            f"{len(COMPARISON_PLANETS)}",
            flush=True,
        )

        print(
            f"DATASET READY: {elapsed:.3f} seconds",
            flush=True,
        )

        print(
            "=" * 70 + "\n",
            flush=True,
        )

    except Exception as exc:

        EXOPLANET_DF = pd.DataFrame()

        TOTAL_OBJECTS = 0
        CONFIRMED_COUNT = 0
        CANDIDATE_COUNT = 0
        FALSE_POSITIVE_COUNT = 0

        DATASET_ERROR = str(exc)

        SEARCH_COLUMNS = []

        COMPARISON_DATA = {}
        COMPARISON_PLANETS = []

        VISUALIZATION_DATA = {
            "disposition_labels": [],
            "disposition_values": [],
            "radius_data": [],
            "period_data": [],
            "temperature_data": [],
            "scatter_period": [],
            "scatter_radius": [],
            "stellar_temperature": [],
            "stellar_radius": [],
        }

        print(
            f"DATASET ERROR: {DATASET_ERROR}",
            flush=True,
        )


# ============================================================
# VISUALIZATION DATA
# ============================================================

def prepare_visualization_data(df):

    data = {
        "disposition_labels": [],
        "disposition_values": [],
        "radius_data": [],
        "period_data": [],
        "temperature_data": [],
        "scatter_period": [],
        "scatter_radius": [],
        "stellar_temperature": [],
        "stellar_radius": [],
    }

    if df.empty:
        return data

    # --------------------------------------------------------
    # 1. Discovery status
    # --------------------------------------------------------

    if "koi_disposition" in df.columns:

        counts = (
            df["koi_disposition"]
            .replace("", "UNKNOWN")
            .fillna("UNKNOWN")
            .value_counts()
        )

        data["disposition_labels"] = [
            str(label)
            for label in counts.index.tolist()
        ]

        data["disposition_values"] = [
            int(value)
            for value in counts.values.tolist()
        ]

    # --------------------------------------------------------
    # 2. Planet radius
    # --------------------------------------------------------

    radius = _prepare_numeric_series(
        df,
        "koi_prad",
    )

    radius = _sample_series(radius)

    data["radius_data"] = [
        float(value)
        for value in radius.tolist()
    ]

    # --------------------------------------------------------
    # 3. Orbital period
    # --------------------------------------------------------

    period = _prepare_numeric_series(
        df,
        "koi_period",
    )

    period = _sample_series(period)

    data["period_data"] = [
        float(value)
        for value in period.tolist()
    ]

    # --------------------------------------------------------
    # 4. Equilibrium temperature
    # --------------------------------------------------------

    temperature = _prepare_numeric_series(
        df,
        "koi_teq",
    )

    temperature = _sample_series(
        temperature
    )

    data["temperature_data"] = [
        float(value)
        for value in temperature.tolist()
    ]

    # --------------------------------------------------------
    # 5. Orbital period vs planet radius
    # --------------------------------------------------------

    if {
        "koi_period",
        "koi_prad",
    }.issubset(df.columns):

        scatter = df[
            [
                "koi_period",
                "koi_prad",
            ]
        ].copy()

        scatter["koi_period"] = pd.to_numeric(
            scatter["koi_period"],
            errors="coerce",
        )

        scatter["koi_prad"] = pd.to_numeric(
            scatter["koi_prad"],
            errors="coerce",
        )

        scatter = scatter.dropna()

        scatter = scatter[
            np.isfinite(
                scatter["koi_period"]
            )
            & np.isfinite(
                scatter["koi_prad"]
            )
            & (
                scatter["koi_period"] > 0
            )
            & (
                scatter["koi_prad"] > 0
            )
        ]

        scatter = _sample_series(
            scatter,
            MAX_SCATTER_POINTS,
        )

        data["scatter_period"] = [
            float(value)
            for value in scatter[
                "koi_period"
            ].tolist()
        ]

        data["scatter_radius"] = [
            float(value)
            for value in scatter[
                "koi_prad"
            ].tolist()
        ]

    # --------------------------------------------------------
    # 6. Stellar temperature vs stellar radius
    # --------------------------------------------------------

    if {
        "koi_steff",
        "koi_srad",
    }.issubset(df.columns):

        stellar = df[
            [
                "koi_steff",
                "koi_srad",
            ]
        ].copy()

        stellar["koi_steff"] = pd.to_numeric(
            stellar["koi_steff"],
            errors="coerce",
        )

        stellar["koi_srad"] = pd.to_numeric(
            stellar["koi_srad"],
            errors="coerce",
        )

        stellar = stellar.dropna()

        stellar = stellar[
            np.isfinite(
                stellar["koi_steff"]
            )
            & np.isfinite(
                stellar["koi_srad"]
            )
            & (
                stellar["koi_steff"] > 0
            )
            & (
                stellar["koi_srad"] > 0
            )
        ]

        stellar = _sample_series(
            stellar,
            MAX_SCATTER_POINTS,
        )

        data["stellar_temperature"] = [
            float(value)
            for value in stellar[
                "koi_steff"
            ].tolist()
        ]

        data["stellar_radius"] = [
            float(value)
            for value in stellar[
                "koi_srad"
            ].tolist()
        ]

    return data


# ============================================================
# LOAD DATASET
# ============================================================

load_dataset()


# ============================================================
# COMMON CONTEXT
# ============================================================

def common_context():

    return {
        "total_objects": TOTAL_OBJECTS,
        "confirmed": CONFIRMED_COUNT,
        "candidates": CANDIDATE_COUNT,
        "false_positive": FALSE_POSITIVE_COUNT,
    }


# ============================================================
# HOME
# ============================================================

def home(request):

    return render(
        request,
        "exoplanets/home.html",
        common_context(),
    )


# ============================================================
# NUMERIC SEARCH
# ============================================================

_NUMERIC_QUERY_RE = re.compile(
    r"^\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)"
    r"\s*"
    r"(>=|<=|>|<|=)?"
    r"\s*"
    r"(-?\d+(?:\.\d+)?)"
    r"(?:\s*-\s*(-?\d+(?:\.\d+)?))?"
    r"\s*$"
)


def apply_numeric_filter(df, query):

    match = _NUMERIC_QUERY_RE.match(
        query
    )

    if not match:
        return None

    column = match.group(1)
    operator = match.group(2)

    first = match.group(3)
    second = match.group(4)

    if column not in df.columns:
        return None

    if not pd.api.types.is_numeric_dtype(
        df[column]
    ):
        return None

    try:

        first = float(first)

        # ----------------------------------------------------
        # Range
        # Example:
        # koi_period 10-100
        # ----------------------------------------------------

        if second is not None:

            second = float(second)

            low = min(
                first,
                second,
            )

            high = max(
                first,
                second,
            )

            return df[
                df[column].between(
                    low,
                    high,
                    inclusive="both",
                )
            ]

        # ----------------------------------------------------
        # Exact value
        # ----------------------------------------------------

        if operator in (
            None,
            "=",
        ):

            return df[
                df[column] == first
            ]

        if operator == ">":

            return df[
                df[column] > first
            ]

        if operator == ">=":

            return df[
                df[column] >= first
            ]

        if operator == "<":

            return df[
                df[column] < first
            ]

        if operator == "<=":

            return df[
                df[column] <= first
            ]

    except (
        ValueError,
        TypeError,
    ):
        return None

    return None


# ============================================================
# RANGE FILTER
# ============================================================

def apply_range(
    dataframe,
    column,
    minimum,
    maximum,
):

    if column not in dataframe.columns:
        return dataframe

    result = dataframe

    if minimum:

        try:

            result = result[
                result[column]
                >= float(minimum)
            ]

        except (
            ValueError,
            TypeError,
        ):
            pass

    if maximum:

        try:

            result = result[
                result[column]
                <= float(maximum)
            ]

        except (
            ValueError,
            TypeError,
        ):
            pass

    return result


# ============================================================
# EXPLORER
# ============================================================

def explorer(request):

    start = time.perf_counter()

    base_context = common_context()

    # --------------------------------------------------------
    # Dataset error
    # --------------------------------------------------------

    if DATASET_ERROR:

        base_context.update({

            "error": (
                f"Dataset loading error: "
                f"{DATASET_ERROR}"
            ),

            "columns": [
                column
                for column in DISPLAY_COLUMNS
                if column in DISPLAY_COLUMNS
            ],

            "page_obj": None,

            "filtered_count": 0,

            "search_query": "",

            "selected_disposition": "",

            "radius_type": "",

            "temperature_type": "",

            "radius_min": "",
            "radius_max": "",

            "teq_min": "",
            "teq_max": "",

            "period_min": "",
            "period_max": "",

            "mag_min": "",
            "mag_max": "",

            "suggestions": DISPLAY_COLUMNS,
        })

        return render(
            request,
            "exoplanets/explorer.html",
            base_context,
        )

    # --------------------------------------------------------
    # GET parameters
    # --------------------------------------------------------

    search_query = request.GET.get(
        "search",
        "",
    ).strip()

    disposition = request.GET.get(
        "disposition",
        "",
    ).strip().upper()

    radius_type = request.GET.get(
        "radius_type",
        "",
    ).strip()

    temperature_type = request.GET.get(
        "temperature_type",
        "",
    ).strip()

    radius_min = request.GET.get(
        "radius_min",
        "",
    ).strip()

    radius_max = request.GET.get(
        "radius_max",
        "",
    ).strip()

    teq_min = request.GET.get(
        "teq_min",
        "",
    ).strip()

    teq_max = request.GET.get(
        "teq_max",
        "",
    ).strip()

    period_min = request.GET.get(
        "period_min",
        "",
    ).strip()

    period_max = request.GET.get(
        "period_max",
        "",
    ).strip()

    mag_min = request.GET.get(
        "mag_min",
        "",
    ).strip()

    mag_max = request.GET.get(
        "mag_max",
        "",
    ).strip()

    # --------------------------------------------------------
    # Work on dataframe reference
    # --------------------------------------------------------

    df = EXOPLANET_DF

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    if search_query:

        numeric_result = apply_numeric_filter(
            df,
            search_query,
        )

        if numeric_result is not None:

            df = numeric_result

        else:

            search_lower = (
                search_query.lower()
            )

            if "_search_text" in df.columns:

                mask = (
                    df["_search_text"]
                    .str.contains(
                        search_lower,
                        regex=False,
                        na=False,
                    )
                )

                df = df.loc[mask]

    # --------------------------------------------------------
    # Disposition
    # --------------------------------------------------------

    if (
        disposition
        and
        "koi_disposition"
        in df.columns
    ):

        df = df.loc[
            df["koi_disposition"]
            == disposition
        ]

    # --------------------------------------------------------
    # Radius category
    # --------------------------------------------------------

    if "koi_prad" in df.columns:

        if radius_type == "small":

            df = df.loc[
                df["koi_prad"] < 1
            ]

        elif radius_type == "medium":

            df = df.loc[
                df["koi_prad"].between(
                    1,
                    2,
                )
            ]

        elif radius_type == "large":

            df = df.loc[
                df["koi_prad"] > 2
            ]

    # --------------------------------------------------------
    # Temperature category
    # --------------------------------------------------------

    if "koi_teq" in df.columns:

        if temperature_type == "cold":

            df = df.loc[
                df["koi_teq"] < 300
            ]

        elif temperature_type == "habitable":

            df = df.loc[
                df["koi_teq"].between(
                    300,
                    500,
                )
            ]

        elif temperature_type == "hot":

            df = df.loc[
                df["koi_teq"] > 500
            ]

    # --------------------------------------------------------
    # Numeric ranges
    # --------------------------------------------------------

    df = apply_range(
        df,
        "koi_prad",
        radius_min,
        radius_max,
    )

    df = apply_range(
        df,
        "koi_teq",
        teq_min,
        teq_max,
    )

    df = apply_range(
        df,
        "koi_period",
        period_min,
        period_max,
    )

    df = apply_range(
        df,
        "koi_kepmag",
        mag_min,
        mag_max,
    )

    filtered_count = int(
        len(df)
    )

    # --------------------------------------------------------
    # Table columns
    # --------------------------------------------------------

    table_columns = [
        column
        for column in DISPLAY_COLUMNS
        if column in df.columns
    ]

    # --------------------------------------------------------
    # Page
    # --------------------------------------------------------

    try:

        requested_page = int(
            request.GET.get(
                "page",
                "1",
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        requested_page = 1

    requested_page = max(
        1,
        requested_page,
    )

    paginator = Paginator(
        range(filtered_count),
        PAGE_SIZE,
    )

    page_obj = paginator.get_page(
        requested_page
    )

    # --------------------------------------------------------
    # Visible page
    # --------------------------------------------------------

    page_indexes = list(
        page_obj.object_list
    )

    if page_indexes:

        page_df = df.iloc[
            page_indexes
        ][table_columns]

    else:

        page_df = df.iloc[
            0:0
        ][table_columns]

    # --------------------------------------------------------
    # Convert page rows
    # --------------------------------------------------------

    table_rows = []

    for values in page_df.itertuples(
        index=False,
        name=None,
    ):

        row = []

        for value in values:

            if value is None:

                row.append("—")
                continue

            try:

                if pd.isna(value):

                    row.append("—")
                    continue

            except (
                TypeError,
                ValueError,
            ):

                pass

            if hasattr(value, "item"):

                try:

                    value = value.item()

                except Exception:

                    pass

            if isinstance(value, float):

                if np.isfinite(value):

                    value = round(
                        value,
                        4,
                    )

                else:

                    value = "—"

            row.append(value)

        table_rows.append(row)

    page_obj.object_list = table_rows

    # --------------------------------------------------------
    # Suggestions
    # --------------------------------------------------------

    if search_query:

        search_lower = (
            search_query.lower()
        )

        suggestions = [
            column
            for column in SEARCH_COLUMNS
            if search_lower
            in column.lower()
        ]

    else:

        suggestions = SEARCH_COLUMNS

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context = {

        **base_context,

        "error": None,

        "columns": table_columns,

        "page_obj": page_obj,

        "filtered_count": filtered_count,

        "search_query": search_query,

        "suggestions": suggestions,

        "selected_disposition": disposition,

        "radius_type": radius_type,

        "radius_min": radius_min,

        "radius_max": radius_max,

        "temperature_type": temperature_type,

        "teq_min": teq_min,

        "teq_max": teq_max,

        "period_min": period_min,

        "period_max": period_max,

        "mag_min": mag_min,

        "mag_max": mag_max,
    }

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"EXPLORER: {filtered_count} records, "
        f"{elapsed:.4f}s",
        flush=True,
    )

    return render(
        request,
        "exoplanets/explorer.html",
        context,
    )


# ============================================================
# VISUALIZATIONS
# ============================================================

def visualizations(request):

    try:

        if DATASET_ERROR:
            raise RuntimeError(
                DATASET_ERROR
            )

        context = {

            "total_objects":
                TOTAL_OBJECTS,

            "confirmed":
                CONFIRMED_COUNT,

            "candidates":
                CANDIDATE_COUNT,

            "false_positive":
                FALSE_POSITIVE_COUNT,

            "visualization_data":
                VISUALIZATION_DATA,

            "error":
                None,
        }

        return render(
            request,
            "exoplanets/visualizations.html",
            context,
        )

    except Exception as exc:

        print(
            "=" * 60,
            flush=True,
        )

        print(
            "VISUALIZATION ERROR:",
            exc,
            flush=True,
        )

        print(
            "=" * 60,
            flush=True,
        )

        empty_visualization_data = {

            "disposition_labels": [],
            "disposition_values": [],
            "radius_data": [],
            "period_data": [],
            "temperature_data": [],
            "scatter_period": [],
            "scatter_radius": [],
            "stellar_temperature": [],
            "stellar_radius": [],
        }

        return render(
            request,
            "exoplanets/visualizations.html",
            {
                "total_objects": 0,
                "confirmed": 0,
                "candidates": 0,
                "false_positive": 0,
                "visualization_data":
                    empty_visualization_data,
                "error": str(exc),
            },
        )


# ============================================================
# EXOPLANET COMPARISON
# ============================================================

def comparison(request):
    """
    Exoplanet Comparison page.

    The comparison dataset is prepared once during
    application startup instead of rebuilding it
    for every request.
    """

    # --------------------------------------------------------
    # Dataset unavailable
    # --------------------------------------------------------

    if DATASET_ERROR or EXOPLANET_DF.empty:

        return render(
            request,
            "exoplanets/comparison.html",
            {
                **common_context(),

                "planets": [],

                "comparison_data": {},

                "selected_planet1": "",

                "selected_planet2": "",

                "error": (
                    DATASET_ERROR
                    or
                    "Exoplanet dataset is unavailable."
                ),
            },
        )

    # --------------------------------------------------------
    # Read selected planets
    # --------------------------------------------------------

    selected_planet1 = request.GET.get(
        "planet1",
        "",
    ).strip()

    selected_planet2 = request.GET.get(
        "planet2",
        "",
    ).strip()

    # --------------------------------------------------------
    # Validate selections
    # --------------------------------------------------------

    if (
        selected_planet1
        not in COMPARISON_DATA
    ):

        selected_planet1 = ""

    if (
        selected_planet2
        not in COMPARISON_DATA
    ):

        selected_planet2 = ""

    # --------------------------------------------------------
    # Build selected comparison
    # --------------------------------------------------------

    selected_comparison = None

    if (
        selected_planet1
        and
        selected_planet2
        and
        selected_planet1 != selected_planet2
    ):

        planet1 = COMPARISON_DATA[
            selected_planet1
        ]

        planet2 = COMPARISON_DATA[
            selected_planet2
        ]

        selected_comparison = (
            build_selected_comparison(
                planet1,
                planet2,
            )
        )

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context = {

        **common_context(),

        "planets":
            COMPARISON_PLANETS,

        "comparison_data":
            COMPARISON_DATA,

        "selected_planet1":
            selected_planet1,

        "selected_planet2":
            selected_planet2,

        "selected_comparison":
            selected_comparison,

        "error":
            None,
    }

    return render(
        request,
        "exoplanets/comparison.html",
        context,
    )

def kepler_summary(request):
    return render(request, "exoplanets/kepler_summary.html")