"""

ARGO Data Platform - Streamlit Dashboard

Main dashboard application with interactive visualizations, chat interface,
and data exploration tools for ARGO oceanographic data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import datetime, date, timedelta
import logging
from typing import Dict, List, Any, Optional


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ARGO Data Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Add padding to main container */
    .main > div {
        padding-top: 2rem;
    }

    /* Metrics styling */
    .stMetric {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        padding: 1rem;
        border-radius: 0.5rem;
    }

    /* Chat message common styles */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        word-wrap: break-word;
    }

    /* User message */
    .user-message {
        background-color: var(--primary-color);
        color: var(--white);
        margin-left: 20%;
    }

    /* Assistant message */
    .assistant-message {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        margin-right: 20%;
        border: 1px solid var(--border-color);
    }

    /* Optional: smooth transition between themes */
    .stMetric, .chat-message, .user-message, .assistant-message {
        transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)



@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_sample_data():
    """Load sample ARGO data for demonstration"""
    # Generate sample data for demonstration
    np.random.seed(42)
    n_profiles = 1000
    
    # Generate realistic ARGO data
    profiles = []
    for i in range(n_profiles):
        # Random coordinates (global distribution)
        lat = np.random.uniform(-60, 70)
        lon = np.random.uniform(-180, 180)
        
        # Random date in last 2 years
        days_ago = np.random.randint(0, 730)
        profile_date = datetime.now() - timedelta(days=days_ago)
        
        # Platform number
        platform = f"590{np.random.randint(1000, 9999)}"
        
        # Cycle number
        cycle = np.random.randint(1, 200)
        
        # Depth range
        max_depth = np.random.uniform(1000, 2000)
        min_depth = 5
        
        # Parameters available (some profiles have BGC, others don't)
        has_bgc = np.random.random() > 0.7
        parameters = ['TEMP', 'PSAL', 'PRES']
        if has_bgc:
            parameters.extend(['DOXY', 'CHLA'])
        
        profiles.append({
            'profile_id': f"{platform}_{cycle}",
            'platform_number': platform,
            'cycle_number': cycle,
            'latitude': lat,
            'longitude': lon,
            'date': profile_date,
            'depth_min': min_depth,
            'depth_max': max_depth,
            'parameters': parameters,
            'has_bgc': has_bgc,
            'region': determine_region(lat, lon)
        })
    
    return pd.DataFrame(profiles)


@st.cache_data(ttl=300)
def load_measurement_data(profile_id: str):
    """Load sample measurement data for a specific profile"""
    np.random.seed(hash(profile_id) % 1000)
    
    # Generate depth levels
    depths = np.arange(5, 2000, 20)
    n_levels = len(depths)
    
    # Generate realistic temperature profile (decreasing with depth)
    temp_surface = np.random.uniform(15, 28)
    temp_deep = np.random.uniform(2, 8)
    temperature = temp_surface * np.exp(-depths/1000) + temp_deep
    temperature += np.random.normal(0, 0.5, n_levels)
    
    # Generate realistic salinity profile
    salinity = np.random.uniform(34, 37, n_levels)
    salinity += np.random.normal(0, 0.2, n_levels)
    
    # Generate oxygen profile (if BGC)
    oxygen = 300 * np.exp(-depths/500) + 50 + np.random.normal(0, 10, n_levels)
    oxygen = np.maximum(oxygen, 0)
    
    # Generate chlorophyll profile (if BGC)
    chlorophyll = np.random.exponential(0.5, n_levels)
    chlorophyll[depths > 200] *= 0.1  # Much lower at depth
    
    return pd.DataFrame({
        'depth': depths,
        'pressure': depths * 1.02,  # Approximate conversion
        'temperature': temperature,
        'salinity': salinity,
        'oxygen': oxygen,
        'chlorophyll': chlorophyll
    })


def determine_region(lat: float, lon: float) -> str:
    """Determine ocean region from coordinates"""
    if lat >= 70:
        return "Arctic"
    elif lat <= -50:
        return "Southern"
    elif 30 <= lat <= 46 and -6 <= lon <= 42:
        return "Mediterranean"
    elif 10 <= lat <= 30 and 50 <= lon <= 80:
        return "Arabian Sea"
    elif 5 <= lat <= 25 and 80 <= lon <= 100:
        return "Bay of Bengal"
    elif lat >= 20 and (120 <= lon <= 240 or -240 <= lon <= -120):
        return "North Pacific"
    elif lat < 0 and (120 <= lon <= 280 or -240 <= lon <= -80):
        return "South Pacific"
    elif lat >= 40 and -80 <= lon <= 0:
        return "North Atlantic"
    elif lat < 0 and -70 <= lon <= 20:
        return "South Atlantic"
    elif -60 <= lat <= 30 and 20 <= lon <= 120:
        return "Indian"
    else:
        return "Other"


def create_map(df: pd.DataFrame) -> folium.Map:
    """Create interactive map with ARGO float locations"""
    
    # Create base map
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=2,
        tiles='OpenStreetMap'
    )
    
    # Add markers for profiles
    for _, row in df.iterrows():
        # Color by region
        region_colors = {
            'North Atlantic': 'red',
            'South Atlantic': 'darkred',
            'North Pacific': 'blue',
            'South Pacific': 'darkblue',
            'Indian': 'green',
            'Arctic': 'white',
            'Southern': 'gray',
            'Mediterranean': 'orange',
            'Arabian Sea': 'purple',
            'Bay of Bengal': 'pink',
            'Other': 'black'
        }
        
        color = region_colors.get(row['region'], 'black')
        
        popup_text = f"""
        Profile: {row['profile_id']}<br>
        Platform: {row['platform_number']}<br>
        Date: {row['date'].strftime('%Y-%m-%d')}<br>
        Region: {row['region']}<br>
        Depth: {row['depth_min']:.0f}-{row['depth_max']:.0f}m<br>
        Parameters: {', '.join(row['parameters'])}
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            popup=popup_text,
            color=color,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    return m


def plot_profile_comparison(df: pd.DataFrame, parameter: str):
    """Create profile comparison plot"""
    
    fig = go.Figure()
    
    # Sample a few profiles for comparison
    sample_profiles = df.sample(min(5, len(df)))
    
    for _, profile in sample_profiles.iterrows():
        # Load measurement data
        measurements = load_measurement_data(profile['profile_id'])
        
        if parameter in measurements.columns:
            fig.add_trace(go.Scatter(
                x=measurements[parameter],
                y=measurements['depth'],
                mode='lines+markers',
                name=f"{profile['platform_number']} ({profile['region']})",
                line=dict(width=2),
                marker=dict(size=3)
            ))
    
    fig.update_layout(
        title=f"{parameter.title()} Profiles Comparison",
        xaxis_title=parameter.title(),
        yaxis_title="Depth (m)",
        yaxis=dict(autorange="reversed"),  # Depth increases downward
        height=500,
        template="plotly_white"
    )
    
    return fig


def plot_geographic_distribution(df, marker_size=5):
    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        hover_name="profile_id",
        hover_data=["region", "date", "depth_max"],
        color="region",
        size_max=marker_size,
        zoom=2,
        height=500
    )
    fig.update_layout(mapbox_style="open-street-map")
    return fig



def plot_temporal_distribution(df: pd.DataFrame):
    """Plot temporal distribution of profiles"""
    
    # Group by month
    df['month'] = df['date'].dt.to_period('M')
    monthly_counts = df.groupby('month').size()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_counts.index.astype(str),
        y=monthly_counts.values,
        mode='lines+markers',
        name='Profiles per month',
        line=dict(width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Temporal Distribution of ARGO Profiles",
        xaxis_title="Month",
        yaxis_title="Number of Profiles",
        height=400,
        template="plotly_white"
    )
    
    return fig


def plot_parameter_availability(df: pd.DataFrame):
    """Plot parameter availability statistics"""
    
    # Count parameter availability
    all_params = []
    for params in df['parameters']:
        all_params.extend(params)
    
    param_counts = pd.Series(all_params).value_counts()
    
    fig = px.bar(
        x=param_counts.index,
        y=param_counts.values,
        title="Parameter Availability in ARGO Profiles",
        labels={'x': 'Parameter', 'y': 'Number of Profiles'}
    )
    
    fig.update_layout(height=400, template="plotly_white")
    return fig


def main():
    """Main dashboard application"""
    
    st.title("🌊 ARGO Data Platform")
    st.markdown("Interactive dashboard for exploring ARGO oceanographic data")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Data Explorer", "Profile Viewer", "Chat Interface", "Data Processing"]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_sample_data()
    
    if page == "Overview":
        show_overview(df)
    elif page == "Data Explorer":
        show_data_explorer(df)
    elif page == "Profile Viewer":
        show_profile_viewer(df)
    elif page == "Chat Interface":
        show_chat_interface()
    elif page == "Data Processing":
        show_data_processing()


def show_overview(df: pd.DataFrame):
    """Show overview dashboard"""
    
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Profiles", len(df))
    
    with col2:
        unique_floats = df['platform_number'].nunique()
        st.metric("Active Floats", unique_floats)
    
    with col3:
        bgc_profiles = df['has_bgc'].sum()
        st.metric("BGC Profiles", bgc_profiles)
    
    with col4:
        latest_date = df['date'].max()
        days_since = (datetime.now() - latest_date).days
        st.metric("Latest Data", f"{days_since} days ago")
    
    # Geographic distribution
    st.subheader("🗺️ Global Distribution")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Interactive map
        m = create_map(df.head(200))  # Limit to 200 points for performance
        st_folium(m, height=400, width=700)
    
    with col2:
        # Region statistics
        region_stats = df['region'].value_counts()
        fig_regions = px.pie(
            values=region_stats.values,
            names=region_stats.index,
            title="Profiles by Region"
        )
        fig_regions.update_layout(height=400)
        st.plotly_chart(fig_regions, use_container_width=True)
    
    # Temporal and parameter distributions
    col1, col2 = st.columns(2)
    
    with col1:
        fig_temporal = plot_temporal_distribution(df)
        st.plotly_chart(fig_temporal, use_container_width=True)
    
    with col2:
        fig_params = plot_parameter_availability(df)
        st.plotly_chart(fig_params, use_container_width=True)

def show_data_explorer(df: pd.DataFrame):
    """Show data explorer interface"""
    
    st.header("🔍 Data Explorer")
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        regions = ['All'] + sorted(df['region'].unique())
        selected_region = st.selectbox("Region", regions)
    
    with col2:
        date_range = st.date_input(
            "Date Range",
            value=(df['date'].min().date(), df['date'].max().date()),
            min_value=df['date'].min().date(),
            max_value=df['date'].max().date()
        )
    
    with col3:
        parameters = ['All', 'Core Only', 'BGC Only']
        param_filter = st.selectbox("Parameter Type", parameters)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['region'] == selected_region]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) &
            (filtered_df['date'].dt.date <= end_date)
        ]
    
    if param_filter == 'BGC Only':
        filtered_df = filtered_df[filtered_df['has_bgc'] == True]
    elif param_filter == 'Core Only':
        filtered_df = filtered_df[filtered_df['has_bgc'] == False]
    
    st.write(f"Showing {len(filtered_df)} profiles (filtered from {len(df)})")
    
    # Geographic plot
    if len(filtered_df) > 0:
        # Modified plotting to increase dot size
        fig_map = plot_geographic_distribution(filtered_df, marker_size=10)  # Add marker_size argument
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Data table
        st.subheader("Profile Data")
        st.dataframe(
            filtered_df[['profile_id', 'platform_number', 'latitude', 'longitude', 
                        'date', 'region', 'depth_max', 'parameters']].head(100),
            use_container_width=True
        )
    else:
        st.warning("No profiles match the selected filters.")

def show_profile_viewer(df: pd.DataFrame):
    """Show individual profile viewer"""
    
    st.header("📈 Profile Viewer")
    
    # Profile selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_profile_id = st.selectbox(
            "Select Profile",
            df['profile_id'].tolist(),
            key="profile_selector"
        )
    
    with col2:
        parameter = st.selectbox(
            "Parameter to View",
            ["temperature", "salinity", "oxygen", "chlorophyll"]
        )
    
    if selected_profile_id:
        # Get profile info
        profile_info = df[df['profile_id'] == selected_profile_id].iloc[0]
        
        # Show profile metadata
        st.subheader("Profile Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Platform:** {profile_info['platform_number']}")
            st.write(f"**Cycle:** {profile_info['cycle_number']}")
            st.write(f"**Date:** {profile_info['date'].strftime('%Y-%m-%d')}")
        
        with col2:
            st.write(f"**Location:** {profile_info['latitude']:.2f}°N, {profile_info['longitude']:.2f}°E")
            st.write(f"**Region:** {profile_info['region']}")
            st.write(f"**Depth Range:** {profile_info['depth_min']:.0f}-{profile_info['depth_max']:.0f}m")
        
        with col3:
            st.write(f"**Parameters:** {', '.join(profile_info['parameters'])}")
            st.write(f"**BGC Data:** {'Yes' if profile_info['has_bgc'] else 'No'}")
        
        # Load and display measurement data
        measurements = load_measurement_data(selected_profile_id)
        
        # Single profile plot
        if parameter in measurements.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=measurements[parameter],
                y=measurements['depth'],
                mode='lines+markers',
                name=parameter.title(),
                line=dict(width=3),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title=f"{parameter.title()} Profile - {selected_profile_id}",
                xaxis_title=parameter.title(),
                yaxis_title="Depth (m)",
                yaxis=dict(autorange="reversed"),
                height=600,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Comparison with other profiles
        st.subheader("Profile Comparison")
        comparison_fig = plot_profile_comparison(
            df[df['region'] == profile_info['region']].head(10),
            parameter
        )
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Raw data table
        with st.expander("View Raw Data"):
            st.dataframe(measurements, use_container_width=True)


def show_chat_interface():
    """Show chat interface for natural language queries"""
    
    st.header("💬 Chat Interface")
    st.markdown("Ask questions about ARGO data in natural language!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your ARGO data assistant. You can ask me questions like:\n\n" +
                         "• Show me temperature profiles near the equator\n" +
                         "• Find salinity data from the Arabian Sea in 2023\n" +
                         "• What are the oxygen levels in the North Atlantic?\n" +
                         "• Compare chlorophyll between different regions\n\n" +
                         "What would you like to know?"
            }
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(
                f'<div class="chat-message user-message">{content}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-message assistant-message">{content}</div>',
                unsafe_allow_html=True
            )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about ARGO data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        st.markdown(
            f'<div class="chat-message user-message">{prompt}</div>',
            unsafe_allow_html=True
        )
        
        # Generate response (placeholder - would use RAG system in production)
        with st.spinner("Thinking..."):
            response = generate_mock_response(prompt)
        
        # Add and display assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(
            f'<div class="chat-message assistant-message">{response}</div>',
            unsafe_allow_html=True
        )
        
        st.rerun()
    
    # Query suggestions
    st.subheader("💡 Try these questions:")
    suggestions = [
        "Show me temperature profiles near the equator",
        "Find salinity measurements in the Arabian Sea for 2023",
        "What are the oxygen levels in the North Atlantic?",
        "Compare chlorophyll concentrations between seasons",
        "Show BGC parameters from the Mediterranean Sea"
    ]
    
    for suggestion in suggestions:
        if st.button(suggestion, key=f"suggestion_{hash(suggestion)}"):
            st.session_state.messages.append({"role": "user", "content": suggestion})
            response = generate_mock_response(suggestion)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()


def generate_mock_response(query: str) -> str:
    """Generate mock response for demonstration"""
    query_lower = query.lower()
    
    if "temperature" in query_lower and "equator" in query_lower:
        return """I found temperature profiles near the equator (±5° latitude). Here's what I discovered:

**SQL Query Generated:**
```sql
SELECT ap.profile_id, ap.latitude, ap.longitude, ap.date,
       AVG(am.temperature) as avg_temp
FROM argo_profiles ap 
JOIN argo_measurements am ON ap.profile_id = am.profile_id
WHERE ap.latitude BETWEEN -5 AND 5 
AND am.temperature IS NOT NULL
GROUP BY ap.profile_id, ap.latitude, ap.longitude, ap.date
ORDER BY ap.date DESC
LIMIT 100;
```

**Results Summary:**
- Found 156 profiles near the equator
- Average surface temperature: 26.8°C
- Temperature range: 23.1°C to 29.4°C
- Most recent data: 15 days ago

The data shows typical tropical ocean temperatures with minimal seasonal variation due to the equatorial location."""
    
    elif "salinity" in query_lower and "arabian" in query_lower:
        return """Here are salinity measurements from the Arabian Sea region:

**SQL Query Generated:**
```sql
SELECT ap.profile_id, ap.latitude, ap.longitude, ap.date,
       am.depth, am.salinity
FROM argo_profiles ap 
JOIN argo_measurements am ON ap.profile_id = am.profile_id
WHERE ap.latitude BETWEEN 10 AND 30 
AND ap.longitude BETWEEN 50 AND 80
AND ap.date >= '2023-01-01'
AND am.salinity IS NOT NULL
ORDER BY ap.date DESC;
```

**Key Findings:**
- 89 profiles from Arabian Sea in 2023
- Surface salinity: 35.8-36.9 PSU (high due to evaporation)
- Depth-averaged salinity: 35.2 PSU
- Notable seasonal variation with monsoon influence

The Arabian Sea shows characteristically high salinity values due to high evaporation rates and limited freshwater input."""
    
    elif "oxygen" in query_lower and "atlantic" in query_lower:
        return """Oxygen level analysis for the North Atlantic:

**SQL Query Generated:**
```sql
SELECT ap.profile_id, ap.latitude, ap.longitude, ap.date,
       am.depth, am.oxygen, am.oxygen_qc
FROM argo_profiles ap 
JOIN argo_measurements am ON ap.profile_id = am.profile_id
WHERE ap.latitude > 40 AND ap.longitude BETWEEN -80 AND 0
AND am.oxygen IS NOT NULL AND am.oxygen_qc IN (1, 2)
ORDER BY am.depth, ap.date DESC;
```

**Oxygen Profile Summary:**
- 234 BGC profiles with oxygen data
- Surface oxygen: 280-320 μmol/kg (good saturation)
- Minimum oxygen zone: 800-1200m depth (~180 μmol/kg)
- Deep water oxygen: 240-280 μmol/kg

North Atlantic waters show good oxygenation due to active deep water formation and mixing processes."""
    
    else:
        return f"""I understand you're asking about: "{query}"

This is a demonstration version of the chat interface. In the full system, I would:

1. **Parse your query** using natural language processing
2. **Search relevant data** using vector similarity
3. **Generate SQL queries** to retrieve specific data
4. **Execute queries** against the ARGO database
5. **Analyze results** and provide insights
6. **Create visualizations** if appropriate

The actual system integrates with:
- PostgreSQL database with ARGO profiles and measurements
- Vector database for semantic search
- LLM for natural language to SQL translation
- Plotly for dynamic visualizations

Would you like to see the data processing or technical details instead?"""


def show_data_processing():
    """Show data processing interface"""
    
    st.header("⚙️ Data Processing")
    
    # Processing stats (mock data)
    st.subheader("Processing Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Files Processed", "1,247", "23 today")
    
    with col2:
        st.metric("Profiles Ingested", "45,892", "456 today")
    
    with col3:
        st.metric("Vector Embeddings", "45,892", "456 today")
    
    with col4:
        st.metric("Last Update", "2 hours ago", "")
    
    # File upload interface
    st.subheader("Upload NetCDF Files")
    uploaded_files = st.file_uploader(
        "Choose ARGO NetCDF files",
        accept_multiple_files=True,
        type=['nc']
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} files for processing:")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size} bytes)")
        
        if st.button("Process Files", type="primary"):
            # Mock processing
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {file.name}...")
                
                # Simulate processing time
                import time
                time.sleep(1)
            
            st.success("All files processed successfully!")
            status_text.text("Processing complete.")
    
    # Recent processing logs
    st.subheader("Recent Processing Activity")
    
    # Mock processing log data
    log_data = pd.DataFrame({
        'Timestamp': pd.date_range(end=pd.Timestamp.now(), periods=10, freq='H')[::-1],
        'Operation': ['File Ingestion'] * 10,
        'Status': ['Completed'] * 8 + ['Failed'] * 2,
        'Files': np.random.randint(1, 20, 10),
        'Profiles': np.random.randint(10, 500, 10),
        'Duration': [f"{np.random.randint(1, 30)} min" for _ in range(10)]
    })
    
    st.dataframe(log_data, use_container_width=True)
    
    # Database health
    st.subheader("Database Status")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("PostgreSQL", "🟢 Healthy", "Response: 45ms")
        st.metric("Vector Store", "🟢 Healthy", "45,892 documents")
    
    with col2:
        st.metric("Disk Usage", "234 GB", "12% increase")
        st.metric("Memory Usage", "8.2 GB", "Normal")


if __name__ == "__main__":
    main()
