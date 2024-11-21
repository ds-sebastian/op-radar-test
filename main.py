import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gc
from typing import Dict, List, Optional

# Define event colors
EVENT_COLORS = {
    "car_on_left": "rgba(255, 0, 0, 0.1)",  # Light red
    "changing_to_left_lane": "rgba(0, 255, 0, 0.1)",  # Light green
    "at_intersections_close_l_to_r": "rgba(0, 0, 255, 0.1)",  # Light blue
    "at_intersections_far_r_to_l": "rgba(255, 165, 0, 0.1)",  # Light orange
}

EVENT_NAMES = {
    "car_on_left": "Car on left (coming towards)",
    "changing_to_left_lane": "Changing to left lane",
    "at_intersections_close_l_to_r": "At intersections (L→R)",
    "at_intersections_far_r_to_l": "Further away cars (R→L)",
}

# Constants
REQUIRED_COLUMNS = [
    "addr",
    "time",
    "LONG_DIST",
    "SPEED",
    "LAT_DIST",
    "LAT_SPEED",
    "RCS",
    "event",
]
MAX_POINTS_PER_PLOT = 10000
CACHE_TTL = 3600  # 1 hour

# Set page configuration
st.set_page_config(layout="wide")


@st.cache_data(ttl=CACHE_TTL)
def load_and_process_data() -> pd.DataFrame:
    """Load data from parquet file with optimization for memory usage"""
    df = pd.read_parquet(
        "preprocessed_full.parquet", engine="pyarrow", columns=REQUIRED_COLUMNS
    )
    return df


@st.cache_data
def get_unique_addresses(data: pd.DataFrame) -> List[str]:
    """Get sorted unique addresses from DataFrame"""
    return sorted(data["addr"].unique())


@st.cache_data
def get_addr_data(data: pd.DataFrame, selected_addr: str) -> pd.DataFrame:
    """Filter data for selected address"""
    return data[data["addr"] == selected_addr].copy()


@st.cache_data
def get_event_shapes(x0: float, x1: float, event_name: str) -> Optional[Dict]:
    """Generate shape dictionary for event highlighting"""
    if event_name not in EVENT_COLORS:
        return None

    return dict(
        type="rect",
        xref="x",
        yref="paper",
        x0=x0,
        x1=x1,
        y0=0,
        y1=1,
        fillcolor=EVENT_COLORS[event_name],
        opacity=0.5,
        layer="below",
        line_width=0,
    )


@st.cache_data
def downsample_data(data: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Downsample data if it exceeds max_points"""
    if len(data) > max_points:
        step = len(data) // max_points
        return data.iloc[::step].copy()
    return data


@st.fragment
def create_plot(data: pd.DataFrame, column: str, container: st.container):
    """Create an optimized plot as a fragment"""
    with container:
        st.subheader(f"{column} over Time")

        # Downsample data if necessary
        plot_data = downsample_data(data, MAX_POINTS_PER_PLOT)

        fig = go.Figure()

        # Add main data trace
        fig.add_trace(
            go.Scatter(
                x=plot_data["time"],
                y=plot_data[column],
                mode="lines",
                name=column,
                line=dict(width=2),
                showlegend=False,
            )
        )

        shapes = []
        events_data = plot_data.dropna(subset=["event"])
        unique_events = events_data["event"].unique()

        # Create shapes for event regions
        for event_name in unique_events:
            event_data = events_data[events_data["event"] == event_name]
            if event_name in EVENT_COLORS:
                x0 = event_data["time"].min()
                x1 = event_data["time"].max()

                shape = get_event_shapes(x0, x1, event_name)
                if shape:
                    shapes.append(shape)

                    # Add reference trace for the legend
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            name=EVENT_NAMES[event_name],
                            fill="none",
                            fillcolor=EVENT_COLORS[event_name],
                            showlegend=True,
                            line=dict(
                                color=EVENT_COLORS[event_name].replace("0.1", "1")
                            ),
                        )
                    )

        # Update layout with optimized settings
        fig.update_layout(
            shapes=shapes,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=0, r=0, t=30, b=0, pad=0),
            xaxis_title="Time (s)",
            yaxis_title=column,
            height=300,
            # Optimize rendering
            uirevision=True,  # Preserve UI state on updates
            hovermode="x unified",  # More efficient hover mode
        )

        # Display the plot with streamlit's native theme
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")


@st.cache_data
def calculate_frequency(data: pd.DataFrame) -> int:
    """Calculate the average frequency of measurements in Hz"""
    time_diffs = data["time"].diff().dropna()
    if len(time_diffs) > 0:
        avg_period = time_diffs.mean()
        return round(1 / avg_period) if avg_period > 0 else 0
    return 0


def sync_query_params(addresses: List[str]):
    """Sync address selection with URL parameters"""
    params = st.query_params
    if "address" in params:
        addr = params["address"]
        if addr in addresses:
            st.session_state.address_index = addresses.index(addr)
    else:
        params["address"] = addresses[st.session_state.address_index]


def create_navigation(addresses: List[str]) -> str:
    """Create optimized navigation controls"""
    nav_container = st.container()
    total_addresses = len(addresses)

    with nav_container:
        prev_col, select_col, next_col = st.columns([1, 2, 1])

        with prev_col:
            prev_disabled = st.session_state.address_index <= 0
            if st.button(
                "← Previous",
                key="prev_btn",
                use_container_width=True,
                disabled=prev_disabled,
            ):
                st.session_state.address_index = max(
                    0, st.session_state.address_index - 1
                )
                # Update query params before rerun
                st.query_params["address"] = addresses[st.session_state.address_index]
                st.rerun()

        with select_col:
            selected_addr = st.selectbox(
                "Select Address",
                addresses,
                index=st.session_state.address_index,
                key="addr_select",
                label_visibility="collapsed",
            )
            # Update index when selectbox changes
            if addresses.index(selected_addr) != st.session_state.address_index:
                st.session_state.address_index = addresses.index(selected_addr)
                st.query_params["address"] = selected_addr
                st.rerun()

        with next_col:
            next_disabled = st.session_state.address_index >= total_addresses - 1
            if st.button(
                "Next →",
                key="next_btn",
                use_container_width=True,
                disabled=next_disabled,
            ):
                st.session_state.address_index = min(
                    total_addresses - 1, st.session_state.address_index + 1
                )
                # Update query params before rerun
                st.query_params["address"] = addresses[st.session_state.address_index]
                st.rerun()

    return selected_addr


def display_metrics(addr_data: pd.DataFrame, frequency: int):
    """Display metrics in a container"""
    st.subheader(f"Address: {addr_data['addr'].iloc[0]}")

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("Total Points", len(addr_data))
    with metrics_col2:
        st.metric(
            "Time Range",
            f"{addr_data['time'].min():.1f}s - {addr_data['time'].max():.1f}s",
        )
    with metrics_col3:
        st.metric("Frequency", f"{frequency} Hz")


def main():
    st.title("Continental Radar Data Visualization - 2023 RAV4")

    # Load the data
    try:
        data = load_and_process_data()
        addresses = get_unique_addresses(data)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    # Initialize session state
    if "address_index" not in st.session_state:
        st.session_state.address_index = 0

    # Sync with query parameters
    sync_query_params(addresses)

    # Create navigation
    selected_addr = create_navigation(addresses)

    # Update address index
    st.session_state.address_index = addresses.index(selected_addr)

    # Filter data for selected address
    addr_data = get_addr_data(data, selected_addr)

    # Calculate and display metrics
    frequency = calculate_frequency(addr_data)
    display_metrics(addr_data, frequency)

    # Create plots
    main_container = st.container()
    with main_container:
        # First row: LONG_DIST and SPEED
        row1_col1, row1_col2 = st.columns(2)
        create_plot(addr_data, "LONG_DIST", row1_col1)
        create_plot(addr_data, "SPEED", row1_col2)

        # Second row: LAT_DIST and LAT_SPEED
        row2_col1, row2_col2 = st.columns(2)
        create_plot(addr_data, "LAT_DIST", row2_col1)
        create_plot(addr_data, "LAT_SPEED", row2_col2)

        # Third row: RCS (full width)
        create_plot(addr_data, "RCS", st.container())

    # Clean up memory
    gc.collect()


if __name__ == "__main__":
    main()
