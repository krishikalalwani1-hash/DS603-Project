
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Spotify Song Clustering",
    page_icon="🎵",
    layout="wide"
)

DATA_DIR = Path(__file__).parent

FILES = {
    "clusters": "spotify_clusters.csv",
    "scaled": "spotify_scaled_features.csv",
    "cleaned": "spotify_cleaned.csv"
}

FEATURES = [
    "danceability",
    "energy",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "valence",
    "tempo",
    "loudness"
]

UNIT_FEATURES = [
    "danceability",
    "energy",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "valence"
]

LABELS = {
    "danceability": "Danceability",
    "energy": "Energy",
    "acousticness": "Acousticness",
    "instrumentalness": "Instrumentalness",
    "liveness": "Liveness",
    "speechiness": "Speechiness",
    "valence": "Positivity (Valence)",
    "tempo": "Tempo",
    "loudness": "Loudness",
    "popularity": "Popularity"
}

FEATURE_DESCRIPTIONS = {
    "danceability": "Dance-oriented",
    "energy": "High-energy",
    "acousticness": "Acoustic",
    "instrumentalness": "Instrumental",
    "liveness": "Live-sounding",
    "speechiness": "Speech-forward",
    "valence": "Upbeat",
    "tempo": "Fast-paced",
    "loudness": "Loud"
}

PALETTE = [
    "#1DB954", "#8F7CF7", "#C6F432",
    "#F0679E", "#25D0C0", "#F6C945",
    "#B9F6CA", "#FF8A3D", "#5FD38D",
    "#5AC8FA"
]

PANEL = "#181818"

GREEN_SCALE = [
    [0, "#0B6E3A"],
    [0.5, "#1DB954"],
    [1, "#B9F6CA"]
]

DIVERGING = [
    [0, "#8F7CF7"],
    [0.5, "#2B2B2B"],
    [1, "#1ED760"]
]


# ============================================================
# 2. PROFESSIONAL CSS
# ============================================================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(
            ellipse at 12% 0%,
            rgba(29,185,84,.075),
            transparent 34%
        ),
        #080b09;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1480px;
}

h1, h2, h3 {
    letter-spacing: -0.03em;
}

h2 {
    font-weight: 760;
}

footer {
    visibility: hidden;
}

[data-testid="stHeaderActionElements"],
.stHeading a,
a.header-anchor,
a[data-testid="stHeaderAnchor"],
a.anchorjs-link,
.anchorjs-link,
h1 a, h2 a, h3 a,
button[aria-label*="Copy link"] {
    display: none !important;
}

.hero {
    background: linear-gradient(
        115deg,
        #173b29 0%,
        #105b34 52%,
        #09251b 100%
    );
    border: 1px solid rgba(91,255,151,.20);
    border-radius: 22px;
    padding: 34px 38px;
    margin: 4px 0 24px;
    box-shadow: 0 16px 45px rgba(0,0,0,.22);
}

.hero-kicker {
    color: #b7f7ce;
    text-transform: uppercase;
    letter-spacing: .19em;
    font-size: .72rem;
    font-weight: 800;
    margin-bottom: 9px;
}

.hero h1 {
    color: white;
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 850;
    margin: 0;
}

.hero p {
    color: rgba(255,255,255,.82);
    font-size: 1rem;
    margin-top: 10px;
    max-width: 750px;
}

.hero-pill {
    display: inline-block;
    margin-top: 17px;
    padding: 6px 11px;
    border: 1px solid rgba(255,255,255,.22);
    border-radius: 999px;
    color: #eafff0;
    font-size: .75rem;
    font-weight: 650;
}

[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        rgba(29,185,84,.13),
        rgba(255,255,255,.018)
    );
    border: 1px solid rgba(29,185,84,.20);
    border-radius: 15px;
    padding: 16px 18px;
    min-height: 105px;
}

.insight-card {
    background: linear-gradient(
        145deg,
        rgba(29,185,84,.10),
        rgba(255,255,255,.018)
    );
    border: 1px solid rgba(29,185,84,.18);
    border-radius: 14px;
    padding: 15px 17px;
    margin: 10px 0 22px;
}

.insight-label {
    color: #8ee6ad;
    text-transform: uppercase;
    letter-spacing: .12em;
    font-size: .68rem;
    font-weight: 800;
    margin-bottom: 7px;
}

.insight-text {
    color: #e4eee7;
    font-size: .91rem;
    line-height: 1.65;
}

.section-note {
    color: #aab7af;
    font-size: .88rem;
    line-height: 1.55;
    margin: -5px 0 13px;
}

.cluster-identity {
    background: linear-gradient(
        110deg,
        rgba(29,185,84,.16),
        rgba(255,255,255,.025)
    );
    border: 1px solid rgba(29,185,84,.30);
    border-radius: 15px;
    padding: 18px 20px;
    margin: 8px 0 20px;
}

.cluster-identity-number {
    color: #9be8b5;
    font-size: .78rem;
    text-transform: uppercase;
    letter-spacing: .12em;
    font-weight: 800;
}

.cluster-identity-name {
    color: white;
    font-size: 1.35rem;
    font-weight: 800;
    margin-top: 5px;
}

.cluster-identity-description {
    color: #c4d3c9;
    font-size: .88rem;
    margin-top: 6px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101b14, #0b100d 60%);
    border-right: 1px solid rgba(255,255,255,.06);
}

.stButton > button {
    background: #1db954;
    color: #07130b;
    border: 0;
    border-radius: 999px;
    font-weight: 750;
}

.stButton > button:hover {
    background: #1ed760;
    color: black;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 13px;
    overflow: hidden;
}

hr {
    border-color: rgba(255,255,255,.09);
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. REUSABLE FUNCTIONS
# ============================================================

def insight_card(label, message):
    """Display an interpretation below a chart."""

    st.markdown(
        dedent(f"""
        <div class="insight-card">
            <div class="insight-label">
                GRAPH INTERPRETATION · {label}
            </div>
            <div class="insight-text">
                {message}
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )


def show_chart(fig, three_d=False):
    """Apply a consistent dark theme to Plotly charts."""

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"color": "#EDEDED"},
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "itemsizing": "constant"
        }
    )

    if three_d:
        axis_style = {
            "backgroundcolor": PANEL,
            "gridcolor": "#333333",
            "zerolinecolor": "#333333"
        }

        fig.update_layout(
            scene={
                "bgcolor": PANEL,
                "xaxis": axis_style,
                "yaxis": axis_style,
                "zaxis": axis_style,
                "dragmode": "orbit"
            }
        )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
            "displayModeBar": "hover",
            "modeBarButtonsToRemove": [
                "lasso2d",
                "select2d"
            ]
        }
    )


def rgba(hex_color, alpha=0.3):
    hex_color = hex_color.lstrip("#")
    r, g, b = (
        int(hex_color[i:i + 2], 16)
        for i in (0, 2, 4)
    )
    return f"rgba({r},{g},{b},{alpha})"


def make_cluster_names(profile):
    """
    Create descriptive cluster names using standardized
    audio-feature profiles.

    Positive profile values mean above-average feature values.
    Negative values mean below-average feature values.
    """

    names = {}
    descriptions = {}

    for cluster_id, row in profile.iterrows():

        available = [
            f for f in FEATURES
            if f in row.index and pd.notna(row[f])
        ]

        if not available:
            names[int(cluster_id)] = f"Cluster {cluster_id}"
            descriptions[int(cluster_id)] = (
                "Audio profile could not be described."
            )
            continue

        sorted_features = row[available].sort_values(
            ascending=False
        )

        strongest_positive = sorted_features.iloc[0]
        positive_feature = sorted_features.index[0]

        if strongest_positive > 0.15:
            main_description = FEATURE_DESCRIPTIONS.get(
                positive_feature,
                LABELS.get(positive_feature, positive_feature)
            )
        else:
            main_description = "Balanced audio profile"

        negative_features = sorted_features[
            sorted_features < -0.35
        ]

        if len(negative_features) > 0:
            negative_feature = negative_features.index[-1]

            negative_description = {
                "energy": "Lower-energy",
                "danceability": "Less dance-oriented",
                "acousticness": "Less acoustic",
                "instrumentalness": "Less instrumental",
                "liveness": "Less live-sounding",
                "speechiness": "Less speech-focused",
                "valence": "Less positive",
                "tempo": "Slower-paced",
                "loudness": "Softer"
            }.get(
                negative_feature,
                f"Lower {LABELS.get(negative_feature, negative_feature)}"
            )

            description = (
                f"{main_description} · {negative_description}"
            )
        else:
            description = main_description

        names[int(cluster_id)] = (
            f"Cluster {cluster_id} · {description}"
        )

        descriptions[int(cluster_id)] = (
            f"This cluster has a relatively high "
            f"{LABELS.get(positive_feature, positive_feature)} "
            f"profile compared with the overall dataset. "
            f"{description}."
        )

    return names, descriptions


def make_colour_map(names, order):
    return {
        names[c]: PALETTE[i % len(PALETTE)]
        for i, c in enumerate(order)
    }


# ============================================================
# 4. LOAD DATA
# ============================================================

@st.cache_data(show_spinner="Loading Spotify dataset...")
def load_data():

    clusters = pd.read_csv(
        DATA_DIR / FILES["clusters"]
    )

    scaled_raw = pd.read_csv(
        DATA_DIR / FILES["scaled"]
    )

    cleaned = pd.read_csv(
        DATA_DIR / FILES["cleaned"],
        low_memory=False
    )

    if not (
        len(clusters) == len(scaled_raw) == len(cleaned)
    ):
        raise ValueError(
            "The three CSV files have different row counts. "
            "Regenerate all three files together."
        )

    # Check row alignment when track_id is available.
    if "track_id" in clusters and "track_id" in cleaned:

        if not np.array_equal(
            clusters["track_id"].astype(str).to_numpy(),
            cleaned["track_id"].astype(str).to_numpy()
        ):
            raise ValueError(
                "The row order in spotify_clusters.csv and "
                "spotify_cleaned.csv does not match."
            )

    songs = clusters.copy()

    songs["cluster"] = pd.to_numeric(
        songs["cluster"],
        errors="raise"
    ).astype(int)

    # Add required columns from cleaned data.
    for column in [
        "track_name",
        "artists",
        "track_genre",
        "popularity"
    ] + FEATURES:

        if column in cleaned.columns:
            songs[column] = cleaned[column].to_numpy()

    # Ensure numeric feature values.
    for feature in FEATURES + ["popularity"]:
        if feature in songs.columns:
            songs[feature] = pd.to_numeric(
                songs[feature],
                errors="coerce"
            )

    # Use only the expected scaled feature columns.
    # This avoids accidental use of an exported index column.
    available_scaled = [
        f for f in FEATURES
        if f in scaled_raw.columns
    ]

    if len(available_scaled) < 2:
        # If scaled CSV has no feature headers, use numeric columns.
        numeric_scaled = scaled_raw.select_dtypes(
            include=np.number
        )
        scaled = numeric_scaled.copy()
    else:
        scaled = scaled_raw[available_scaled].copy()

    scaled = scaled.apply(
        pd.to_numeric,
        errors="coerce"
    )

    scaled = scaled.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if scaled.isna().any().any():
        scaled = scaled.fillna(
            scaled.mean()
        ).fillna(0)

    if len(scaled.columns) < 2:
        raise ValueError(
            "The scaled feature file must contain at least "
            "two numeric feature columns."
        )

    # Build cluster profiles from standardized features.
    profile = scaled.groupby(
        songs["cluster"].to_numpy()
    ).mean()

    profile.index.name = "cluster"

    names, descriptions = make_cluster_names(profile)

    songs["cluster_name"] = songs["cluster"].map(names)

    return songs, scaled, profile, names, descriptions


@st.cache_data(show_spinner="Calculating PCA...")
def calculate_pca():

    _, scaled, _, _, _ = load_data()

    pca = PCA(
        n_components=3,
        random_state=42
    )

    coordinates = pca.fit_transform(
        scaled.to_numpy()
    )

    return coordinates, pca.explained_variance_ratio_


@st.cache_data(show_spinner="Calculating silhouette score...")
def calculate_silhouette():

    songs, scaled, _, _, _ = load_data()

    labels = songs["cluster"].to_numpy()

    if len(np.unique(labels)) < 2:
        return np.nan

    sample_size = min(10000, len(songs))

    return silhouette_score(
        scaled.to_numpy(),
        labels,
        sample_size=sample_size,
        random_state=42
    )


@st.cache_resource(show_spinner="Preparing song similarity search...")
def get_neighbour_model():

    _, scaled, _, _, _ = load_data()

    model = NearestNeighbors(
        n_neighbors=min(51, len(scaled)),
        metric="euclidean"
    )

    model.fit(scaled.to_numpy())

    return model


# ============================================================
# 5. 3D ROTATION CONTROLS
# ============================================================

def rotate_vector(vector, axis, angle):

    axis = np.asarray(axis, dtype=float)
    vector = np.asarray(vector, dtype=float)

    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1 - np.cos(angle))
    )


def add_rotation_controls(fig):

    initial_eye = np.array([1.6, 1.6, 0.9])
    initial_up = np.array([0.0, 0.0, 1.0])

    axes = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0])
    }

    frames = []
    buttons = []
    steps = 48

    for axis_name, axis_vector in axes.items():

        frame_names = []

        for step in range(steps + 1):

            angle = 2 * np.pi * step / steps

            eye = rotate_vector(
                initial_eye,
                axis_vector,
                angle
            )

            up = rotate_vector(
                initial_up,
                axis_vector,
                angle
            )

            frame_name = f"rotate_{axis_name}_{step}"
            frame_names.append(frame_name)

            frames.append(
                go.Frame(
                    name=frame_name,
                    layout=go.Layout(
                        scene_camera={
                            "eye": {
                                "x": float(eye[0]),
                                "y": float(eye[1]),
                                "z": float(eye[2])
                            },
                            "up": {
                                "x": float(up[0]),
                                "y": float(up[1]),
                                "z": float(up[2])
                            }
                        }
                    )
                )
            )

        buttons.append({
            "label": f"360° {axis_name}",
            "method": "animate",
            "args": [
                frame_names,
                {
                    "frame": {
                        "duration": 45,
                        "redraw": False
                    },
                    "transition": {"duration": 0},
                    "fromcurrent": True,
                    "mode": "immediate"
                }
            ]
        })

    buttons.append({
        "label": "Stop",
        "method": "animate",
        "args": [
            [None],
            {
                "frame": {"duration": 0},
                "transition": {"duration": 0},
                "mode": "immediate"
            }
        ]
    })

    fig.frames = frames

    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "direction": "right",
            "showactive": False,
            "x": 0.01,
            "y": 1.10,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "#202a23",
            "bordercolor": "#3a5944",
            "borderwidth": 1,
            "font": {
                "color": "#eafff0",
                "size": 11
            },
            "buttons": buttons
        }]
    )

    return fig


# ============================================================
# 6. OVERVIEW TAB
# ============================================================

def tab_overview(songs, profile, names, descriptions, order, colours):

    st.header("Overview")

    total_songs = len(songs)
    total_clusters = songs["cluster"].nunique()
    sizes = songs["cluster"].value_counts()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total songs", f"{total_songs:,}")
    c2.metric("Clusters found", total_clusters)
    c3.metric(
        "Largest cluster",
        f"{100 * sizes.iloc[0] / total_songs:.1f}%"
    )

    silhouette = calculate_silhouette()

    c4.metric(
        "Silhouette score",
        f"{silhouette:.3f}" if pd.notna(silhouette) else "N/A"
    )

    st.caption(
        "The silhouette score measures how closely songs fit "
        "their own cluster compared with other clusters. "
        "A higher score generally indicates better separation."
    )

    # ---------------- PCA SONG MAP ----------------

    st.subheader("Song map · Global structure")

    st.markdown(
        '<div class="section-note">Each point represents one song. '
        'PCA compresses the audio features into two or three dimensions '
        'to help visualize the overall structure.</div>',
        unsafe_allow_html=True
    )

    coordinates, variance = calculate_pca()

    left, right = st.columns([3, 1])

    with right:

        view = st.radio(
            "Chart view",
            ["3D", "2D"],
            horizontal=True
        )

        axes_mode = st.radio(
            "Dimensions",
            ["PCA dimensions", "Select audio features"]
        )

        if axes_mode == "Select audio features":

            x_feature = st.selectbox(
                "X axis",
                FEATURES,
                index=1,
                format_func=lambda x: LABELS[x]
            )

            y_feature = st.selectbox(
                "Y axis",
                FEATURES,
                index=2,
                format_func=lambda x: LABELS[x]
            )

            z_feature = None

            if view == "3D":
                z_feature = st.selectbox(
                    "Z axis",
                    FEATURES,
                    index=6,
                    format_func=lambda x: LABELS[x]
                )

        sample_n = st.slider(
            "Songs to display",
            min_value=min(1000, total_songs),
            max_value=total_songs,
            value=min(6000, total_songs),
            step=500 if total_songs >= 1500 else 1
        )

        if axes_mode == "PCA dimensions":
            number_of_components = 3 if view == "3D" else 2
            st.caption(
                f"Selected components explain "
                f"{variance[:number_of_components].sum() * 100:.1f}% "
                f"of the total variation."
            )

    indices = np.random.default_rng(42).choice(
        total_songs,
        size=sample_n,
        replace=False
    )

    plot_data = songs.iloc[indices].copy()

    if axes_mode == "PCA dimensions":

        plot_data["PC1"] = coordinates[indices, 0]
        plot_data["PC2"] = coordinates[indices, 1]
        plot_data["PC3"] = coordinates[indices, 2]

        x_col, y_col = "PC1", "PC2"
        z_col = "PC3"

        axis_labels = {
            "PC1": f"PC1 ({variance[0] * 100:.1f}%)",
            "PC2": f"PC2 ({variance[1] * 100:.1f}%)",
            "PC3": f"PC3 ({variance[2] * 100:.1f}%)"
        }

    else:

        x_col = x_feature
        y_col = y_feature
        z_col = z_feature

        axis_labels = {
            x_col: LABELS[x_col],
            y_col: LABELS[y_col]
        }

        if z_col:
            axis_labels[z_col] = LABELS[z_col]

    common = {
        "color": "cluster_name",
        "color_discrete_map": colours,
        "category_orders": {
            "cluster_name": [names[c] for c in order]
        },
        "hover_name": "track_name",
        "hover_data": {
            "artists": True,
            "track_genre": True,
            "cluster": True
        },
        "labels": axis_labels,
        "opacity": 0.8
    }

    with left:

        if view == "3D":

            fig = px.scatter_3d(
                plot_data,
                x=x_col,
                y=y_col,
                z=z_col,
                height=620,
                **common
            )

            fig.update_traces(
                marker={"size": 4}
            )

            fig.update_layout(
                legend_title_text="",
                scene={
                    "camera": {
                        "eye": {
                            "x": 1.6,
                            "y": 1.6,
                            "z": 0.9
                        }
                    },
                    "dragmode": "orbit"
                }
            )

            fig = add_rotation_controls(fig)

        else:

            fig = px.scatter(
                plot_data,
                x=x_col,
                y=y_col,
                height=540,
                render_mode="webgl",
                **common
            )

            fig.update_traces(
                marker={"size": 6}
            )

            fig.update_layout(
                legend_title_text=""
            )

        show_chart(
            fig,
            three_d=(view == "3D")
        )

    if axes_mode == "PCA dimensions":

        interpretation = (
            f"The selected {3 if view == '3D' else 2} principal "
            f"components explain "
            f"{variance[:3 if view == '3D' else 2].sum() * 100:.1f}% "
            "of the variation in the audio features. Each point is "
            "a song. Songs located close to each other have more "
            "similar feature profiles in this reduced-dimensional view. "
            "Overlapping groups indicate that some songs have similar "
            "audio characteristics."
        )

    else:

        interpretation = (
            f"This chart compares {LABELS[x_col]} and "
            f"{LABELS[y_col]}"
            + (
                f", along with {LABELS[z_col]}"
                if z_col else ""
            )
            + ". Songs positioned close together have similar values "
            "on the selected axes. This view shows only the selected "
            "features, not every feature used in clustering."
        )

    insight_card("Song map", interpretation)

    # ---------------- CLUSTER SIZE ----------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Cluster sizes")

        size_data = pd.DataFrame({
            "Cluster": [names[c] for c in order],
            "Songs": [int(sizes[c]) for c in order]
        })

        fig = px.bar(
            size_data,
            x="Songs",
            y="Cluster",
            orientation="h",
            color="Cluster",
            color_discrete_map=colours,
            height=400
        )

        fig.update_layout(
            showlegend=False,
            yaxis={"autorange": "reversed", "title": ""}
        )

        show_chart(fig)

        largest = sizes.idxmax()
        smallest = sizes.idxmin()

        insight_card(
            "Cluster size distribution",
            f"{names[largest]} is the largest cluster, with "
            f"{sizes[largest]:,} songs. {names[smallest]} is the "
            f"smallest, with {sizes[smallest]:,} songs. Cluster size "
            "shows how many songs were assigned to each group; it "
            "does not independently measure cluster quality."
        )

    # ---------------- FEATURE HEATMAP ----------------

    with col2:

        st.subheader("What defines each cluster")

        heatmap_data = profile.loc[order].copy()

        heatmap_data.index = [
            names[c] for c in heatmap_data.index
        ]

        heatmap_data = heatmap_data.rename(
            columns=LABELS
        )

        fig = px.imshow(
            heatmap_data,
            aspect="auto",
            color_continuous_scale=DIVERGING,
            zmin=-2,
            zmax=2,
            text_auto=".1f",
            height=400
        )

        fig.update_layout(
            coloraxis_colorbar={"title": "Z-score"}
        )

        show_chart(fig)

        insight_card(
            "Cluster feature heatmap",
            "The heatmap shows the standardized average feature "
            "values for each cluster. Positive values indicate "
            "above-average feature levels, while negative values "
            "indicate below-average levels. Stronger colors indicate "
            "larger differences from the overall dataset average."
        )

    # ---------------- POPULARITY ----------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Popularity by cluster")

        popularity_data = songs.copy()
        popularity_data["Cluster name"] = (
            popularity_data["cluster"].map(names)
        )

        fig = px.box(
            popularity_data,
            x="Cluster name",
            y="popularity",
            color="Cluster name",
            color_discrete_map=colours,
            points=False,
            height=420
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Cluster",
            yaxis_title="Popularity",
            xaxis_tickangle=-35
        )

        show_chart(fig)

        median_popularity = (
            popularity_data.groupby("Cluster name")["popularity"]
            .median()
            .dropna()
            .sort_values()
        )

        if not median_popularity.empty:
            insight_card(
                "Popularity distribution",
                f"{median_popularity.index[-1]} has the highest "
                f"median popularity ({median_popularity.iloc[-1]:.1f}), "
                f"while {median_popularity.index[0]} has the lowest "
                f"({median_popularity.iloc[0]:.1f}). The box plots "
                "compare the median and spread of popularity scores."
            )

    # ---------------- GENRE DISTRIBUTION ----------------

    with col2:

        st.subheader("Most common genres")

        genre_limit = st.slider(
            "Number of genres",
            5,
            20,
            10,
            key="overview_genre_limit"
        )

        genre_data = (
            songs["track_genre"]
            .value_counts()
            .head(genre_limit)
            .rename_axis("Genre")
            .reset_index(name="Songs")
        )

        fig = px.bar(
            genre_data,
            x="Songs",
            y="Genre",
            orientation="h",
            color="Songs",
            color_continuous_scale=GREEN_SCALE,
            height=420
        )

        fig.update_layout(
            yaxis={"autorange": "reversed", "title": ""},
            coloraxis_showscale=False
        )

        show_chart(fig)

        if not genre_data.empty:
            top = genre_data.iloc[0]

            insight_card(
                "Genre frequency",
                f"{top['Genre']} is the most represented genre "
                f"among the displayed genres, with {top['Songs']:,} "
                "songs. This chart shows genre frequency, not "
                "the similarity between songs."
            )


# ============================================================
# 7. CLUSTER EXPLORER TAB
# ============================================================

def tab_cluster_explorer(songs, profile, names, descriptions, order, colours):

    st.header("Cluster Explorer")

    selected_cluster = st.selectbox(
        "Select a cluster",
        order,
        format_func=lambda c: names[c]
    )

    subset = songs[
        songs["cluster"] == selected_cluster
    ].copy()

    cluster_colour = colours[names[selected_cluster]]

    # Full-width cluster identity section.
    # This avoids the truncation caused by st.metric.

    st.markdown(
        dedent(f"""
        <div class="cluster-identity">
            <div class="cluster-identity-number">
                CLUSTER {selected_cluster}
            </div>
            <div class="cluster-identity-name">
                {names[selected_cluster]}
            </div>
            <div class="cluster-identity-description">
                {descriptions[selected_cluster]}
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Songs", f"{len(subset):,}")

    c2.metric(
        "Share of dataset",
        f"{100 * len(subset) / len(songs):.1f}%"
    )

    c3.metric(
        "Average popularity",
        f"{subset['popularity'].mean():.1f}/100"
    )

    # ---------------- AUDIO FINGERPRINT ----------------

    left, right = st.columns(2)

    with left:

        st.subheader("Audio fingerprint")

        cluster_means = [
            float(subset[f].mean())
            for f in UNIT_FEATURES
        ]

        overall_means = [
            float(songs[f].mean())
            for f in UNIT_FEATURES
        ]

        feature_labels = [
            LABELS[f] for f in UNIT_FEATURES
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=cluster_means + [cluster_means[0]],
                theta=feature_labels + [feature_labels[0]],
                fill="toself",
                name=names[selected_cluster],
                line={
                    "color": cluster_colour,
                    "width": 2.5
                },
                fillcolor=rgba(cluster_colour, 0.3)
            )
        )

        fig.add_trace(
            go.Scatterpolar(
                r=overall_means + [overall_means[0]],
                theta=feature_labels + [feature_labels[0]],
                name="Overall average",
                line={
                    "color": "rgba(255,255,255,0.45)",
                    "dash": "dash"
                }
            )
        )

        fig.update_layout(
            polar={
                "bgcolor": PANEL,
                "radialaxis": {
                    "visible": True,
                    "range": [0, 1],
                    "gridcolor": "#333333"
                }
            },
            height=390
        )

        show_chart(fig)

        highest_index = int(np.argmax(cluster_means))
        lowest_index = int(np.argmin(cluster_means))

        insight_card(
            "Audio fingerprint",
            f"The highest average feature in this cluster is "
            f"{feature_labels[highest_index]} "
            f"({cluster_means[highest_index]:.2f}), while the lowest "
            f"is {feature_labels[lowest_index]} "
            f"({cluster_means[lowest_index]:.2f}). The solid area "
            "represents this cluster, and the dashed line represents "
            "the overall dataset average."
        )

    # ---------------- FEATURE DEVIATIONS ----------------

    with right:

        st.subheader("Sound profile deviations")

        deviations = profile.loc[selected_cluster].copy()
        deviations = deviations.rename(index=LABELS).sort_values()

        deviation_data = pd.DataFrame({
            "Feature": deviations.index,
            "Deviation": deviations.values
        })

        deviation_data["Direction"] = np.where(
            deviation_data["Deviation"] >= 0,
            "Above average",
            "Below average"
        )

        fig = px.bar(
            deviation_data,
            x="Deviation",
            y="Feature",
            orientation="h",
            color="Direction",
            color_discrete_map={
                "Above average": "#1DB954",
                "Below average": "#8F7CF7"
            },
            height=390
        )

        show_chart(fig)

        largest_feature = deviations.abs().idxmax()

        insight_card(
            "Feature deviations",
            f"The largest absolute deviation for this cluster "
            f"is {largest_feature}, with a standardized value of "
            f"{deviations[largest_feature]:+.2f}. Positive values "
            "indicate above-average feature levels, while negative "
            "values indicate below-average levels."
        )

    # ---------------- SONG TABLE ----------------

    st.divider()
    st.subheader("Explore songs in this cluster")

    f1, f2, f3 = st.columns([2, 1, 1])

    with f1:
        search_text = st.text_input(
            "Search song or artist",
            placeholder="Type a song title or artist...",
            key="cluster_search"
        )

    with f2:
        sort_option = st.selectbox(
            "Sort by",
            [
                "Popularity",
                "Danceability",
                "Energy",
                "Tempo",
                "Valence"
            ],
            key="cluster_sort"
        )

    with f3:
        rows = st.slider(
            "Rows to display",
            5,
            100,
            25,
            step=5,
            key="cluster_rows"
        )

    filtered = subset.copy()

    if search_text.strip():

        q = search_text.strip()

        match = (
            filtered["track_name"].astype(str).str.contains(
                q,
                case=False,
                regex=False,
                na=False
            )
            |
            filtered["artists"].astype(str).str.contains(
                q,
                case=False,
                regex=False,
                na=False
            )
        )

        filtered = filtered[match]

    sort_columns = {
        "Popularity": "popularity",
        "Danceability": "danceability",
        "Energy": "energy",
        "Tempo": "tempo",
        "Valence": "valence"
    }

    filtered = filtered.sort_values(
        sort_columns[sort_option],
        ascending=False
    ).head(rows)

    display_table = filtered[
        [
            "track_name",
            "artists",
            "track_genre",
            "popularity",
            "danceability",
            "energy",
            "tempo"
        ]
    ].rename(columns={
        "track_name": "Song",
        "artists": "Artist",
        "track_genre": "Genre",
        "popularity": "Popularity",
        "danceability": "Danceability",
        "energy": "Energy",
        "tempo": "Tempo (BPM)"
    })

    st.caption(
        f"Showing {len(display_table)} songs from "
        f"{len(subset):,} songs in this cluster."
    )

    st.dataframe(
        display_table,
        hide_index=True,
        use_container_width=True
    )

    insight_card(
        "Song table",
        f"The table contains songs assigned to {names[selected_cluster]}. "
        f"The current sorting variable is {sort_option}. You can search "
        "by song title or artist to find specific tracks."
    )

    # ---------------- POPULARITY AND GENRE ----------------

    left, right = st.columns(2)

    with left:

        st.subheader("Popularity distribution")

        fig = px.histogram(
            subset,
            x="popularity",
            nbins=20,
            color_discrete_sequence=[cluster_colour],
            height=350
        )

        fig.update_layout(
            xaxis_title="Popularity score",
            yaxis_title="Number of songs"
        )

        show_chart(fig)

        insight_card(
            "Popularity distribution",
            f"The mean popularity is {subset['popularity'].mean():.1f} "
            f"and the median popularity is "
            f"{subset['popularity'].median():.1f}. The histogram "
            "shows how popularity values are distributed within "
            "this cluster."
        )

    with right:

        st.subheader("Genre mix")

        genre_data = (
            subset["track_genre"]
            .value_counts()
            .head(12)
            .rename_axis("Genre")
            .reset_index(name="Songs")
        )

        fig = px.bar(
            genre_data,
            x="Songs",
            y="Genre",
            orientation="h",
            color="Songs",
            color_continuous_scale=GREEN_SCALE,
            height=350
        )

        fig.update_layout(
            yaxis={"autorange": "reversed", "title": ""},
            coloraxis_showscale=False
        )

        show_chart(fig)

        if not genre_data.empty:
            dominant = genre_data.iloc[0]

            insight_card(
                "Genre composition",
                f"The most frequent genre is {dominant['Genre']}, "
                f"with {dominant['Songs']} songs. The chart shows "
                "whether this cluster is dominated by a small number "
                "of genres or contains a wider mixture."
            )


# ============================================================
# 8. SONG EXPLORER TAB
# ============================================================

def tab_song_explorer(songs, names):

    st.header("Song Explorer")

    st.markdown(
        "Search for a song to identify its cluster and find "
        "other tracks with similar audio characteristics."
    )

    query = st.text_input(
        "Search by song name",
        placeholder="e.g. Bohemian Rhapsody, Yellow, Stay"
    )

    if len(query.strip()) < 2:
        st.info("Type at least two letters to search for a song.")
        return

    matches = songs[
        songs["track_name"].astype(str).str.contains(
            query.strip(),
            case=False,
            regex=False,
            na=False
        )
    ].sort_values(
        "popularity",
        ascending=False
    ).head(100)

    if matches.empty:
        st.warning("No matching songs found. Try another title.")
        return

    labels = [
        f"{row.track_name} — {row.artists} ({row.track_genre})"
        for row in matches.itertuples()
    ]

    selected_position = st.selectbox(
        f"{len(matches)} matching songs",
        range(len(matches)),
        format_func=lambda i: labels[i]
    )

    song = matches.iloc[selected_position]

    # Find the actual row position in the original songs dataframe.
    position = songs.index.get_loc(song.name)

    st.subheader(song["track_name"])
    st.caption(
        f"{song['artists']} · {song['track_genre']}"
    )

    cluster_id = int(song["cluster"])
    cluster_name = names[cluster_id]

    # Full-width cluster label instead of st.metric.
    st.markdown(
        dedent(f"""
        <div class="cluster-identity">
            <div class="cluster-identity-number">
                ASSIGNED CLUSTER · {cluster_id}
            </div>
            <div class="cluster-identity-name">
                {cluster_name}
            </div>
            <div class="cluster-identity-description">
                This song belongs to the audio-feature group
                identified by the profile above.
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Popularity",
        f"{song['popularity']:.0f}/100"
    )

    c2.metric(
        "Tempo",
        f"{song['tempo']:.0f} BPM"
    )

    c3.metric(
        "Loudness",
        f"{song['loudness']:.1f} dB"
    )

    # ---------------- SONG AUDIO FEATURES ----------------

    left, right = st.columns(2)

    with left:

        st.subheader("Sound features")

        feature_data = pd.DataFrame({
            "Feature": [
                LABELS[f] for f in UNIT_FEATURES
            ],
            "Value": [
                song[f] for f in UNIT_FEATURES
            ]
        })

        fig = px.bar(
            feature_data,
            x="Value",
            y="Feature",
            orientation="h",
            range_x=[0, 1],
            color="Value",
            color_continuous_scale=GREEN_SCALE,
            range_color=[0, 1],
            height=370
        )

        fig.update_layout(
            yaxis={"autorange": "reversed", "title": ""},
            coloraxis_showscale=False
        )

        show_chart(fig)

        highest = feature_data.loc[
            feature_data["Value"].idxmax()
        ]

        lowest = feature_data.loc[
            feature_data["Value"].idxmin()
        ]

        insight_card(
            "Selected song audio profile",
            f"{highest['Feature']} has the highest value "
            f"({highest['Value']:.2f}), while {lowest['Feature']} "
            f"has the lowest value ({lowest['Value']:.2f}). "
            "These seven values describe the selected song's "
            "audio characteristics."
        )

    # ---------------- NEAREST NEIGHBOURS ----------------

    with right:

        st.subheader("Most similar songs")

        st.markdown(
            '<div class="section-note">Nearest neighbours are songs '
            'with the smallest Euclidean distance across the scaled '
            'audio features.</div>',
            unsafe_allow_html=True
        )

        neighbour_count = st.slider(
            "Number of similar songs",
            3,
            20,
            10
        )

        _, scaled, _, _, _ = load_data()

        distances, indices = get_neighbour_model().kneighbors(
            scaled.iloc[[position]].to_numpy(),
            n_neighbors=min(51, len(songs))
        )

        neighbours = [
            (distance, index)
            for distance, index in zip(
                distances[0],
                indices[0]
            )
            if index != position
        ][:neighbour_count]

        neighbour_rows = songs.iloc[
            [index for _, index in neighbours]
        ]

        neighbour_table = pd.DataFrame({
            "Song": neighbour_rows["track_name"].to_numpy(),
            "Artist": neighbour_rows["artists"].to_numpy(),
            "Genre": neighbour_rows["track_genre"].to_numpy(),
            "Cluster": [
                names[int(c)]
                for c in neighbour_rows["cluster"].to_numpy()
            ],
            "Distance": [
                round(float(distance), 3)
                for distance, _ in neighbours
            ]
        })

        same_cluster = sum(
            int(row["cluster"]) == cluster_id
            for _, row in neighbour_rows.iterrows()
        )

        st.dataframe(
            neighbour_table,
            hide_index=True,
            use_container_width=True
        )

        insight_card(
            "Nearest-neighbour results",
            f"{same_cluster} of the {len(neighbour_table)} nearest "
            "songs belong to the same cluster as the selected song. "
            "Smaller distance values indicate more similar audio "
            "feature profiles in the scaled feature space."
        )

    # ---------------- SONG VS CLUSTER AVERAGE ----------------

    st.subheader("Selected song vs. cluster average")

    cluster_songs = songs[
        songs["cluster"] == cluster_id
    ]

    comparison = pd.DataFrame({
        "Feature": [
            LABELS[f] for f in UNIT_FEATURES
        ],
        "Selected song": [
            float(song[f]) for f in UNIT_FEATURES
        ],
        "Cluster average": [
            float(cluster_songs[f].mean())
            for f in UNIT_FEATURES
        ]
    })

    comparison_long = comparison.melt(
        id_vars="Feature",
        var_name="Profile",
        value_name="Value"
    )

    fig = px.bar(
        comparison_long,
        x="Feature",
        y="Value",
        color="Profile",
        barmode="group",
        color_discrete_map={
            "Selected song": "#1DB954",
            "Cluster average": "#8F7CF7"
        },
        height=390
    )

    fig.update_layout(
        yaxis={"range": [0, 1], "title": "Feature value"},
        xaxis_title="Audio feature"
    )

    show_chart(fig)

    differences = (
        comparison["Selected song"]
        - comparison["Cluster average"]
    )

    largest_difference_index = differences.abs().idxmax()

    insight_card(
        "Song compared with cluster average",
        f"The largest difference occurs in "
        f"{comparison.loc[largest_difference_index, 'Feature']}, "
        f"with a difference of "
        f"{differences.iloc[largest_difference_index]:+.2f}. "
        "Smaller differences indicate that the song is more typical "
        "of its cluster for that feature."
    )


# ============================================================
# 9. COMPARE CLUSTERS TAB
# ============================================================

def tab_compare(songs, profile, names, order, colours):

    st.header("Compare Clusters")

    st.markdown(
        "Select two clusters to compare their average audio "
        "characteristics and standardized feature deviations."
    )

    c1, c2 = st.columns(2)

    with c1:
        first = st.selectbox(
            "First cluster",
            order,
            format_func=lambda c: names[c],
            key="compare_first"
        )

    remaining = [
        c for c in order
        if c != first
    ]

    with c2:
        second = st.selectbox(
            "Second cluster",
            remaining,
            format_func=lambda c: names[c],
            key="compare_second"
        )

    left_data = songs[
        songs["cluster"] == first
    ]

    right_data = songs[
        songs["cluster"] == second
    ]

    popularity_gap = (
        left_data["popularity"].mean()
        - right_data["popularity"].mean()
    )

    tempo_gap = (
        left_data["tempo"].mean()
        - right_data["tempo"].mean()
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("First cluster songs", f"{len(left_data):,}")
    m2.metric("Second cluster songs", f"{len(right_data):,}")
    m3.metric("Popularity difference", f"{popularity_gap:+.1f}")
    m4.metric("Tempo difference", f"{tempo_gap:+.1f} BPM")

    # ---------------- RAW FEATURE COMPARISON ----------------

    comparison = pd.DataFrame({
        "Feature": [LABELS[f] for f in FEATURES],
        names[first]: [
            left_data[f].mean() for f in FEATURES
        ],
        names[second]: [
            right_data[f].mean() for f in FEATURES
        ]
    })

    left, right = st.columns(2)

    with left:

        st.subheader("Average audio features")

        long_data = comparison.melt(
            id_vars="Feature",
            var_name="Cluster",
            value_name="Average feature value"
        )

        fig = px.bar(
            long_data,
            x="Feature",
            y="Average feature value",
            color="Cluster",
            barmode="group",
            color_discrete_map={
                names[first]: colours[names[first]],
                names[second]: colours[names[second]]
            },
            height=430
        )

        fig.update_layout(
            xaxis_tickangle=-35,
            legend_title_text=""
        )

        show_chart(fig)

        differences = (
            comparison.set_index("Feature")[names[first]]
            - comparison.set_index("Feature")[names[second]]
        )

        strongest_raw = differences.abs().idxmax()

        insight_card(
            "Average feature comparison",
            f"The largest difference in raw average values occurs "
            f"in {strongest_raw}. The chart compares the average "
            "audio-feature values of the two selected clusters. "
            "Remember that tempo and loudness use different units "
            "from the 0–1 audio attributes."
        )

    # ---------------- STANDARDIZED COMPARISON ----------------

    with right:

        st.subheader("Deviation from the dataset average")

        means = songs[FEATURES].mean()
        standard_deviations = songs[FEATURES].std().replace(0, 1)

        z_first = (
            left_data[FEATURES].mean() - means
        ) / standard_deviations

        z_second = (
            right_data[FEATURES].mean() - means
        ) / standard_deviations

        z_data = pd.DataFrame({
            "Feature": [LABELS[f] for f in FEATURES],
            names[first]: z_first.to_numpy(),
            names[second]: z_second.to_numpy()
        })

        z_long = z_data.melt(
            id_vars="Feature",
            var_name="Cluster",
            value_name="Standardized deviation"
        )

        fig = px.bar(
            z_long,
            x="Standardized deviation",
            y="Feature",
            color="Cluster",
            barmode="group",
            orientation="h",
            color_discrete_map={
                names[first]: colours[names[first]],
                names[second]: colours[names[second]]
            },
            height=430
        )

        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="#aab7af"
        )

        fig.update_layout(
            legend_title_text=""
        )

        show_chart(fig)

        z_pivot = z_data.set_index("Feature")

        z_difference = (
            z_pivot[names[first]]
            - z_pivot[names[second]]
        )

        strongest_standardized = z_difference.abs().idxmax()

        insight_card(
            "Standardized deviation comparison",
            f"The strongest standardized difference occurs in "
            f"{strongest_standardized}. Positive values indicate "
            "above-average feature levels, while negative values "
            "indicate below-average levels. Standardization makes "
            "features with different units easier to compare."
        )

    st.divider()

    insight_card(
        "Overall comparison",
        f"You are comparing {names[first]} with {names[second]}. "
        f"The mean popularity difference is {popularity_gap:+.1f} "
        f"points, and the mean tempo difference is "
        f"{tempo_gap:+.1f} BPM. These are descriptive differences, "
        "not evidence of causation."
    )


# ============================================================
# 10. DATASET INSIGHTS TAB
# ============================================================

def tab_dataset_insights(songs):

    st.header("Dataset Insights")

    st.markdown(
        "Explore the distributions, correlations and descriptive "
        "statistics of the audio features."
    )

    numeric = songs[
        FEATURES + ["popularity"]
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Audio variables", len(FEATURES))

    c2.metric(
        "Genres represented",
        f"{songs['track_genre'].nunique():,}"
    )

    c3.metric(
        "Complete numerical records",
        f"{numeric.dropna().shape[0]:,}"
    )

    left, right = st.columns(2)

    # ---------------- CORRELATION ----------------

    with left:

        st.subheader("Feature correlation matrix")

        correlation = numeric.corr(method="pearson")

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=[
                [0, "#8F7CF7"],
                [0.5, "#202820"],
                [1, "#1ED760"]
            ],
            zmin=-1,
            zmax=1,
            height=510
        )

        fig.update_layout(
            coloraxis_colorbar={"title": "Pearson r"}
        )

        show_chart(fig)

        upper = correlation.where(
            np.triu(
                np.ones(correlation.shape),
                k=1
            ).astype(bool)
        )

        pairs = upper.abs().stack()

        if not pairs.empty:

            strongest_pair = pairs.idxmax()

            value = correlation.loc[
                strongest_pair[0],
                strongest_pair[1]
            ]

            insight_card(
                "Correlation matrix",
                f"The strongest absolute pairwise correlation is "
                f"between {strongest_pair[0]} and "
                f"{strongest_pair[1]}, with r = {value:.2f}. "
                "Positive correlations indicate that two variables "
                "tend to increase together. Negative correlations "
                "indicate an inverse relationship. Correlation does "
                "not establish causation."
            )

    # ---------------- FEATURE DISTRIBUTION ----------------

    with right:

        st.subheader("Feature distribution")

        selected_feature = st.selectbox(
            "Select a feature",
            FEATURES + ["popularity"],
            format_func=lambda f: LABELS.get(f, f),
            key="dataset_distribution_feature"
        )

        feature_values = numeric[
            selected_feature
        ].dropna()

        fig = px.histogram(
            feature_values,
            x=selected_feature,
            nbins=35,
            marginal="box",
            color_discrete_sequence=["#1DB954"],
            height=510
        )

        fig.update_layout(
            xaxis_title=LABELS.get(
                selected_feature,
                selected_feature
            ),
            yaxis_title="Number of songs"
        )

        show_chart(fig)

        if not feature_values.empty:

            mean_value = feature_values.mean()
            median_value = feature_values.median()
            std_value = feature_values.std()
            skewness = feature_values.skew()

            if skewness > 0.5:
                shape = "positively skewed"
            elif skewness < -0.5:
                shape = "negatively skewed"
            else:
                shape = "approximately symmetric"

            insight_card(
                "Selected feature distribution",
                f"The distribution of "
                f"{LABELS.get(selected_feature, selected_feature)} "
                f"is {shape}. Its mean is {mean_value:.3f}, "
                f"median is {median_value:.3f}, and standard "
                f"deviation is {std_value:.3f}. The histogram "
                "shows the frequency of values, while the box plot "
                "summarizes the median, quartiles and potential "
                "extreme observations."
            )

    # ---------------- DESCRIPTIVE STATISTICS ----------------

    st.divider()
    st.subheader("Descriptive statistics")

    summary = (
        numeric.describe()
        .T
        .reset_index()
        .rename(columns={
            "index": "Variable",
            "count": "N",
            "mean": "Mean",
            "std": "Std. deviation",
            "min": "Minimum",
            "25%": "Q1",
            "50%": "Median",
            "75%": "Q3",
            "max": "Maximum"
        })
    )

    st.dataframe(
        summary.round(3),
        hide_index=True,
        use_container_width=True
    )

    insight_card(
        "Descriptive statistics",
        "The table summarizes the numerical variables using "
        "sample size, mean, standard deviation, minimum, quartiles "
        "and maximum. The mean describes the central value, "
        "standard deviation describes spread, and quartiles "
        "summarize the distribution."
    )


# ============================================================
# 11. MAIN APPLICATION
# ============================================================

def main():

    # ---------------- HERO ----------------

    st.markdown(dedent("""
    <div class="hero">
        <div class="hero-kicker">
            Music data · Unsupervised learning
        </div>

        <h1>Spotify Song Clustering</h1>

        <p>
            An interactive analytics dashboard for examining
            audio-feature similarity, cluster structure,
            and track-level patterns in a Spotify dataset.
        </p>

        <span class="hero-pill">
            CLUSTERING · PCA · AUDIO FEATURE ANALYSIS
        </span>
    </div>
    """), unsafe_allow_html=True)

    # ---------------- CHECK DATA FILES ----------------

    missing_files = [
        filename
        for filename in FILES.values()
        if not (DATA_DIR / filename).exists()
    ]

    if missing_files:

        st.error(
            "The following files are missing from the app folder: "
            + ", ".join(missing_files)
        )

        st.info(
            "Place all three CSV files in the same folder as "
            "this Python file."
        )

        st.stop()

    try:

        songs, scaled, profile, names, descriptions = load_data()

    except Exception as error:

        st.error(
            f"Unable to load the dataset: {error}"
        )

        st.stop()

    order = (
        songs["cluster"]
        .value_counts()
        .index.tolist()
    )

    colours = make_colour_map(
        names,
        order
    )

    # ---------------- SIDEBAR ----------------

    with st.sidebar:

        st.header("Navigation")

        st.markdown(dedent("""
        **Overview**

        Global cluster structure and key indicators.

        **Cluster explorer**

        Inspect a cluster's audio profile and songs.

        **Compare clusters**

        Compare two groups using feature summaries.

        **Song explorer**

        Search for a song and inspect similar tracks.

        **Dataset insights**

        Explore distributions and correlations.
        """))

        st.divider()

        st.markdown("**Dataset snapshot**")

        st.caption(f"{len(songs):,} tracks")

        st.caption(
            f"{songs['track_genre'].nunique():,} genres"
        )

        st.caption(
            f"{songs['cluster'].nunique():,} clusters"
        )

        st.divider()

        st.caption(
            "Cluster names are generated from the average "
            "standardized audio-feature profiles. They are "
            "descriptive labels, not original genre labels."
        )

    # ---------------- TABS ----------------

    overview_tab, cluster_tab, compare_tab, song_tab, insights_tab = (
        st.tabs([
            "Overview",
            "Cluster Explorer",
            "Compare Clusters",
            "Song Explorer",
            "Dataset Insights"
        ])
    )

    with overview_tab:

        tab_overview(
            songs,
            profile,
            names,
            descriptions,
            order,
            colours
        )

    with cluster_tab:

        tab_cluster_explorer(
            songs,
            profile,
            names,
            descriptions,
            order,
            colours
        )

    with compare_tab:

        tab_compare(
            songs,
            profile,
            names,
            order,
            colours
        )

    with song_tab:

        tab_song_explorer(
            songs,
            names
        )

    with insights_tab:

        tab_dataset_insights(songs)


if __name__ == "__main__":
    main()
