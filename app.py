"""
Streamlit Dashboard for Power Procurement Strategy
===================================================
Interactive web UI for power procurement recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Page configuration
st.set_page_config(
    page_title="Power Procurement Strategy Advisor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #0f3460;
    }
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def load_models():
    """Load trained models (with fallback to demo mode)."""
    try:
        from src.forecaster import PowerForecaster
        from src.recommender import ProcurementRecommender
        
        forecaster = PowerForecaster("config.yaml")
        forecaster.load("models/forecaster")
        
        recommender = ProcurementRecommender("config.yaml")
        recommender.load("models/recommender")
        
        return forecaster, recommender, True
    except Exception as e:
        st.warning(f"⚠️ Running in demo mode. Models not loaded: {e}")
        return None, None, False


def generate_demo_prediction(config: dict) -> dict:
    """Generate demo prediction when models aren't available."""
    np.random.seed(42)
    
    # Simulate power trace based on inputs
    t = np.arange(48)
    base = config['num_gpus'] * 0.5  # Base power per GPU
    
    if config['workload_type'] == 'LLM_Training':
        util_profile = 0.9 + np.random.normal(0, 0.05, 48)
    else:
        util_profile = 0.5 + 0.3 * np.sin(2 * np.pi * (t - 6) / 24) + np.random.normal(0, 0.1, 48)
    
    power_trace = base * util_profile * np.random.uniform(0.8, 1.2)
    
    # Simple heuristic for procurement mix
    avg_power = np.mean(power_trace) / 1000  # Convert to MW
    
    ppa_ratio = config['renewable_target_pct'] / 100
    ppa = avg_power * ppa_ratio
    spot = avg_power * (1 - ppa_ratio) * 0.8
    battery = avg_power * 0.1
    
    cost = config['ppa_price'] * ppa_ratio + config['spot_price'] * (1 - ppa_ratio) * 0.7
    
    return {
        'power_trace': power_trace.tolist(),
        'recommended_mix': {'ppa': round(ppa, 3), 'spot': round(spot, 3), 'battery': round(battery, 3)},
        'contract_lead_time_months': int(np.random.randint(6, 18)),
        'forecasted_cost_usd_mwh': round(cost, 2)
    }


def main():
    # Header
    st.markdown('<h1 class="main-header">⚡ Power Procurement Strategy Advisor</h1>', unsafe_allow_html=True)
    st.markdown("*AI-powered recommendations for data center power procurement*")
    
    # Load models
    forecaster, recommender, models_loaded = load_models()
    
    # Sidebar - Input Configuration
    st.sidebar.header("🖥️ Cluster Configuration")
    
    num_gpus = st.sidebar.select_slider(
        "Number of GPUs",
        options=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
        value=512
    )
    
    gpu_type = st.sidebar.selectbox(
        "GPU Type",
        options=["A100", "H100", "MI300X"],
        index=1
    )
    
    workload_type = st.sidebar.selectbox(
        "Workload Type",
        options=["LLM_Training", "Inference"],
        index=0
    )
    
    st.sidebar.header("⚙️ Workload Parameters")
    
    batch_size = st.sidebar.number_input("Batch Size", min_value=1, max_value=512, value=32)
    seq_length = st.sidebar.number_input("Sequence Length", min_value=128, max_value=8192, value=2048)
    
    st.sidebar.header("🏢 Site Constraints")
    
    grid_capacity = st.sidebar.slider("Grid Capacity (MW)", 10, 200, 50)
    renewable_target = st.sidebar.slider("Renewable Target (%)", 0, 100, 80)
    
    st.sidebar.header("💰 Market Prices")
    
    ppa_price = st.sidebar.slider("PPA Price ($/MWh)", 30, 100, 55)
    spot_price = st.sidebar.slider("Spot Price ($/MWh)", 25, 150, 70)
    battery_cost = st.sidebar.slider("Battery Cost ($/kWh)", 100, 400, 200)
    
    # Build configuration
    config = {
        'num_gpus': num_gpus,
        'gpu_type': gpu_type,
        'workload_type': workload_type,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'grid_capacity_mw': grid_capacity,
        'renewable_target_pct': renewable_target,
        'ppa_price': ppa_price,
        'spot_price': spot_price,
        'battery_cost': battery_cost
    }
    
    # Generate Prediction Button
    if st.sidebar.button("🔮 Generate Recommendation", type="primary", use_container_width=True):
        with st.spinner("Analyzing cluster configuration..."):
            prediction = generate_demo_prediction(config)
            st.session_state['prediction'] = prediction
            st.session_state['config'] = config
    
    # Main Content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Power Demand Forecast")
        
        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            power_trace = pred['power_trace']
            
            # Power trace plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(48)),
                y=power_trace,
                mode='lines',
                fill='tozeroy',
                line=dict(color='#667eea', width=2),
                fillcolor='rgba(102, 126, 234, 0.3)',
                name='Power Demand'
            ))
            fig.update_layout(
                title="48-Hour Power Demand Forecast",
                xaxis_title="Hour",
                yaxis_title="Power (kW)",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Avg Power", f"{np.mean(power_trace):.1f} kW")
            col_b.metric("Peak Power", f"{np.max(power_trace):.1f} kW")
            col_c.metric("Min Power", f"{np.min(power_trace):.1f} kW")
            col_d.metric("Std Dev", f"{np.std(power_trace):.1f} kW")
        else:
            st.info("👈 Configure your cluster and click 'Generate Recommendation' to see the forecast")
    
    with col2:
        st.subheader("🎯 Recommended Strategy")
        
        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            mix = pred['recommended_mix']
            
            # Procurement mix metrics
            st.metric("💰 Forecasted Cost", f"${pred['forecasted_cost_usd_mwh']:.2f}/MWh")
            st.metric("📅 Lead Time", f"{pred['contract_lead_time_months']} months")
            
            # Mix pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['PPA', 'Spot', 'Battery'],
                values=[mix['ppa'], mix['spot'], mix['battery']],
                hole=0.4,
                marker_colors=['#2ecc71', '#e74c3c', '#3498db']
            )])
            fig.update_layout(
                title="Procurement Mix (MW)",
                template="plotly_dark",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Mix details
            total = mix['ppa'] + mix['spot'] + mix['battery']
            st.write("**Mix Breakdown:**")
            st.write(f"- 🌿 PPA: {mix['ppa']:.3f} MW ({mix['ppa']/total*100:.1f}%)")
            st.write(f"- ⚡ Spot: {mix['spot']:.3f} MW ({mix['spot']/total*100:.1f}%)")
            st.write(f"- 🔋 Battery: {mix['battery']:.3f} MW ({mix['battery']/total*100:.1f}%)")
    
    # Additional Analysis Section
    if 'prediction' in st.session_state:
        st.divider()
        st.subheader("📈 Detailed Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Cost Breakdown", "Hourly Mix", "Recommendations"])
        
        with tab1:
            pred = st.session_state['prediction']
            config = st.session_state['config']
            mix = pred['recommended_mix']
            
            # Cost breakdown
            ppa_cost = mix['ppa'] * config['ppa_price']
            spot_cost = mix['spot'] * config['spot_price']
            battery_cost = mix['battery'] * config['battery_cost'] * 0.1
            
            fig = go.Figure(data=[go.Bar(
                x=['PPA', 'Spot', 'Battery'],
                y=[ppa_cost, spot_cost, battery_cost],
                marker_color=['#2ecc71', '#e74c3c', '#3498db']
            )])
            fig.update_layout(
                title="Cost Breakdown ($/hour)",
                yaxis_title="Cost ($)",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Simulated hourly procurement
            power_trace = np.array(pred['power_trace'])
            mix = pred['recommended_mix']
            total_mix = mix['ppa'] + mix['spot'] + mix['battery']
            
            ppa_hourly = (mix['ppa'] / total_mix) * power_trace
            spot_hourly = (mix['spot'] / total_mix) * power_trace
            battery_hourly = (mix['battery'] / total_mix) * power_trace
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(48)), y=ppa_hourly, stackgroup='one', name='PPA', fillcolor='rgba(46, 204, 113, 0.7)'))
            fig.add_trace(go.Scatter(x=list(range(48)), y=spot_hourly, stackgroup='one', name='Spot', fillcolor='rgba(231, 76, 60, 0.7)'))
            fig.add_trace(go.Scatter(x=list(range(48)), y=battery_hourly, stackgroup='one', name='Battery', fillcolor='rgba(52, 152, 219, 0.7)'))
            
            fig.update_layout(
                title="Hourly Procurement Mix",
                xaxis_title="Hour",
                yaxis_title="Power (kW)",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("""
            ### 💡 Strategic Recommendations
            
            Based on your configuration, here are key recommendations:
            
            1. **Long-term PPA contracts** should cover your baseload renewable requirements
            2. **Spot market purchases** provide flexibility for variable demand
            3. **Battery storage** helps with peak shaving and arbitrage opportunities
            
            #### Next Steps:
            - Engage with recommended vendors for PPA negotiations
            - Model seasonal variations in demand
            - Consider on-site renewable generation for additional savings
            """)
    
    # Footer
    st.divider()
    st.markdown("*Built with ❤️ for sustainable AI infrastructure*")


if __name__ == "__main__":
    main()
