import streamlit as st
import sys
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="IndicaAI - Pharmaceutical Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    .header-text {
        position: fixed;
        top: 10px;
        right: 20px;
        font-size: 12px;
        color: #666;
        z-index: 999;
    }
    
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 18px;
        margin-bottom: 30px;
    }
    
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .stTextInput > div > div > input {
        font-size: 16px;
        padding: 10px;
    }
    
    .response-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    
    .user-message {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
    }
    
    .model-selector {
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agents_loaded' not in st.session_state:
    st.session_state.agents_loaded = False
if 'agents' not in st.session_state:
    st.session_state.agents = None

# Load pharmaceutical agents (you'll need to import your actual code)
@st.cache_resource
def load_pharmaceutical_agents():
    """Load and cache pharmaceutical agents"""
    try:
        # Import your pharmaceutical intelligence code here
        # For now, we'll create a simple mock that you can replace
        
        class MockAgent:
            def __init__(self, name, response_style):
                self.name = name
                self.response_style = response_style
            
            def answer_query(self, query):
                if self.response_style == "naive":
                    return f"Basic pharmaceutical analysis: {query}\n\nThis is a simple response showing basic information about pharmaceutical topics."
                
                elif self.response_style == "classical":
                    return f"📊 DETAILED PHARMACEUTICAL ANALYSIS\n" + "="*50 + f"\n\nQuery: {query}\n\n• Market Overview: Comprehensive analysis available\n• Competitive Landscape: Multiple players identified\n• Strategic Recommendations: Further investigation recommended\n\nThis represents enhanced market intelligence with structured analysis."
                
                elif self.response_style == "deep":
                    return f"🚀 ADVANCED PHARMACEUTICAL INTELLIGENCE REPORT\n" + "="*60 + f"\n\nExecutive Summary: {query}\n\n🎯 KEY INSIGHTS:\n• Advanced AI-powered analysis\n• Multi-dimensional market assessment\n• Strategic recommendations with risk evaluation\n\n💡 STRATEGIC IMPLICATIONS:\n• Market opportunity identification\n• Competitive positioning analysis\n• Investment recommendations\n\n⚠️ RISK ASSESSMENT:\n• Market volatility considerations\n• Regulatory impact evaluation\n• Competitive threat analysis\n\nThis represents sophisticated pharmaceutical market intelligence."
        
        # Create mock agents (replace with your actual agent initialization)
        agents = {
            'naive_agent': MockAgent("Naive Agent", "naive"),
            'classical_agent': MockAgent("Classical ML Agent", "classical"), 
            'deep_agent': MockAgent("Deep Learning Agent", "deep")
        }
        
        return agents
        
    except Exception as e:
        st.error(f"Error loading pharmaceutical agents: {e}")
        return None

def main_chat_interface():
    """Main chat interface"""
    
    # Header with project info
    st.markdown('<div class="header-text">Project for Evan Moh | Duke University AIPI540</div>', unsafe_allow_html=True)
    
    # Main title
    st.markdown('<h1 class="main-title">IndicaAI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Pharmaceutical Marketing Intelligence Platform</p>', unsafe_allow_html=True)
    
    # Load agents
    if not st.session_state.agents_loaded:
        with st.spinner("🔄 Loading pharmaceutical intelligence models..."):
            agents = load_pharmaceutical_agents()
            if agents:
                st.session_state.agents = agents
                st.session_state.agents_loaded = True
                st.success("✅ Pharmaceutical intelligence models loaded successfully!")
            else:
                st.error("❌ Failed to load pharmaceutical models")
                return
    
    # Model selection
    st.markdown('<div class="model-selector">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        selected_model = st.selectbox(
            "Select AI Model:",
            options=['Deep Learning Agent', 'Classical ML Agent', 'Naive Agent'],
            index=0,
            help="Choose the AI model for pharmaceutical analysis"
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Map selection to agent
    model_mapping = {
        'Deep Learning Agent': 'deep_agent',
        'Classical ML Agent': 'classical_agent', 
        'Naive Agent': 'naive_agent'
    }
    
    # Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="response-container"><strong>{message["model"]}:</strong><br><pre>{message["content"]}</pre></div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_input(
        "Ask me questions regarding pharmaceutical oncology:",
        placeholder="e.g., What are the upcoming oncology launches in 2025?",
        key="user_input"
    )
    
    # Process user input
    if user_input and st.session_state.agents_loaded:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get response from selected model
        with st.spinner(f"🤔 {selected_model} is analyzing your query..."):
            try:
                selected_agent_key = model_mapping[selected_model]
                agent = st.session_state.agents[selected_agent_key]
                
                start_time = time.time()
                response = agent.answer_query(user_input)
                response_time = time.time() - start_time
                
                # Add response to chat
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "model": selected_model,
                    "response_time": response_time
                })
                
                # Clear input and rerun to show new message
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Clear chat button
    if st.session_state.messages:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Chat History", type="secondary"):
                st.session_state.messages = []
                st.experimental_rerun()

def evaluation_metrics_page():
    """Evaluation metrics and model comparison page"""
    
    st.markdown('<div class="header-text">Project for Evan Moh | Duke University AIPI540</div>', unsafe_allow_html=True)
    
    st.title("📊 Model Evaluation & Performance Metrics")
    st.markdown("---")
    
    # Model performance comparison
    st.header("🏆 Model Performance Comparison")
    
    # Sample evaluation data (replace with your actual evaluation results)
    evaluation_data = {
        'Model': ['Deep Learning Agent', 'Classical ML Agent', 'Naive Agent'],
        'Overall Score': [0.457, 0.407, 0.295],
        'Technical Performance': [0.917, 0.900, 0.812],
        'Business Intelligence': [0.525, 0.500, 0.264],
        'Clinical Intelligence': [0.000, 0.000, 0.000],
        'Strategic Support': [0.222, 0.037, 0.000]
    }
    
    df = pd.DataFrame(evaluation_data)
    
    # Overall performance chart
    fig_overall = px.bar(
        df, 
        x='Model', 
        y='Overall Score',
        title="Overall Model Performance",
        color='Overall Score',
        color_continuous_scale='Blues'
    )
    fig_overall.update_layout(showlegend=False)
    st.plotly_chart(fig_overall, use_container_width=True)
    
    # Detailed performance metrics
    st.header("📈 Detailed Performance Breakdown")
    
    # Radar chart for multi-dimensional comparison
    categories = ['Technical Performance', 'Business Intelligence', 'Clinical Intelligence', 'Strategic Support']
    
    fig_radar = go.Figure()
    
    for i, model in enumerate(df['Model']):
        values = [df.iloc[i][cat] for cat in categories]
        values += [values[0]]  # Close the radar chart
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=model
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Multi-Dimensional Performance Comparison"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Performance metrics table
    st.header("📋 Detailed Metrics Table")
    
    # Format percentages
    df_display = df.copy()
    for col in df_display.columns:
        if col != 'Model':
            df_display[col] = df_display[col].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(df_display, use_container_width=True)
    
    # Key insights
    st.header("💡 Key Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Performance Highlights")
        st.write("""
        **Deep Learning Agent (45.7%)**
        - Best overall performance
        - Strongest in technical capabilities
        - Advanced strategic analysis
        
        **Classical ML Agent (40.7%)**
        - Solid business intelligence
        - Good technical performance
        - Moderate strategic insights
        
        **Naive Agent (29.5%)**
        - Basic functionality
        - Limited strategic capabilities
        - Suitable for simple queries
        """)
    
    with col2:
        st.subheader("📈 Improvement Areas")
        st.write("""
        **All Models Need:**
        - Enhanced clinical intelligence
        - Improved safety information
        - Better medical terminology usage
        
        **Recommendations:**
        - Deep Learning Agent: Ready for pilot deployment
        - Classical ML Agent: Good for market analysis
        - Naive Agent: Basic lookup functions only
        """)
    
    # Model capabilities comparison
    st.header("🔍 Model Capabilities Comparison")
    
    capabilities_data = {
        'Capability': ['Response Depth', 'Market Data Access', 'Strategic Insights', 'Clinical Knowledge', 'Use Case'],
        'Naive Agent': ['Basic', 'Limited', 'Minimal', 'None', 'Quick lookups'],
        'Classical ML Agent': ['Detailed', 'Comprehensive', 'Moderate', 'Limited', 'Market analysis'],
        'Deep Learning Agent': ['Sophisticated', 'Full integration', 'Advanced', 'Basic', 'Strategic planning']
    }
    
    capabilities_df = pd.DataFrame(capabilities_data)
    st.dataframe(capabilities_df, use_container_width=True)

# Navigation
def main():
    """Main application with navigation"""
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Select Page:",
        ["💬 Chat Interface", "📊 Evaluation Metrics"]
    )
    
    # Page routing
    if page == "💬 Chat Interface":
        main_chat_interface()
    elif page == "📊 Evaluation Metrics":
        evaluation_metrics_page()

if __name__ == "__main__":
    main()
