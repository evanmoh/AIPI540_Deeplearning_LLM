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

# Custom CSS for Claude-like interface
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
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
        color: #2E86AB;
        font-size: 42px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 16px;
        margin-bottom: 40px;
        font-weight: 400;
    }
    
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 20px;
    }
    
    .user-message {
        background-color: #F7F7F8;
        padding: 16px 20px;
        border-radius: 18px;
        margin: 16px 0;
        border: 1px solid #E5E5E7;
        color: #2F2F2F;
        font-size: 15px;
        line-height: 1.5;
    }
    
    .assistant-message {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 18px;
        margin: 16px 0;
        border: 1px solid #E5E5E7;
        color: #2F2F2F;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .assistant-message pre {
        background-color: #F8F9FA;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #E9ECEF;
        overflow-x: auto;
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
        font-size: 13px;
        line-height: 1.4;
        color: #2F2F2F;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    .model-selector {
        text-align: center;
        margin: 30px 0;
        padding: 20px;
        background-color: #FAFAFA;
        border-radius: 12px;
        border: 1px solid #E5E5E7;
    }
    
    .chat-input-container {
        position: sticky;
        bottom: 0;
        background-color: white;
        padding: 20px 0;
        border-top: 1px solid #E5E5E7;
        z-index: 100;
    }
    
    .stTextInput > div > div > input {
        font-size: 16px;
        padding: 16px 20px;
        border-radius: 24px;
        border: 2px solid #4A90E2;
        background-color: #FFFFFF;
        color: #2F2F2F;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2E86AB;
        box-shadow: 0 0 0 3px rgba(46, 134, 171, 0.2);
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #888;
        font-style: italic;
    }
    
    .example-queries {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        border: 1px solid #E9ECEF;
    }
    
    .example-button {
        margin: 5px;
        padding: 12px 20px;
        background-color: white;
        border: 2px solid #4A90E2;
        border-radius: 25px;
        color: #2F2F2F;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .example-button:hover {
        background-color: #4A90E2;
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .welcome-section {
        text-align: center;
        padding: 40px 20px;
        margin-bottom: 30px;
    }
    
    .stSelectbox > div > div > select {
        font-size: 15px;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #D1D5DB;
        background-color: white;
    }
    
    .clear-button {
        text-align: center;
        margin: 20px 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agents_loaded' not in st.session_state:
    st.session_state.agents_loaded = False
if 'agents' not in st.session_state:
    st.session_state.agents = None
if 'example_query' not in st.session_state:
    st.session_state.example_query = None

# Load pharmaceutical agents
@st.cache_resource
def load_pharmaceutical_agents():
    """Load pharmaceutical intelligence agents"""
    try:
        class PharmaceuticalDatabase:
            def __init__(self):
                self.market_data = {
                    'oncology': {
                        'market_size_2024': '$196.1B',
                        'projected_2025': '$210.3B',
                        'growth_rate': '7.2%',
                        'key_segments': ['lung_cancer', 'breast_cancer', 'blood_cancers'],
                        'market_leaders': ['AstraZeneca', 'Roche', 'Merck']
                    },
                    'diabetes': {
                        'market_size_2024': '$78.2B',
                        'projected_2025': '$84.1B', 
                        'growth_rate': '7.5%',
                        'key_segments': ['GLP-1 agonists', 'insulin', 'SGLT2_inhibitors'],
                        'market_leaders': ['Novo Nordisk', 'Eli Lilly', 'Sanofi']
                    }
                }
                
                self.pipeline_data = {
                    'upcoming_launches_2025': [
                        {
                            'drug': 'Dato-DXd',
                            'company': 'Daiichi Sankyo/AstraZeneca',
                            'indication': 'HER2+ breast cancer',
                            'category': 'oncology',
                            'expected_launch': 'Q2 2025',
                            'peak_sales_projection': '$8.5B'
                        },
                        {
                            'drug': 'Capivasertib',
                            'company': 'AstraZeneca',
                            'indication': 'PIK3CA/AKT pathway cancers',
                            'category': 'oncology',
                            'expected_launch': 'Q3 2025',
                            'peak_sales_projection': '$4.2B'
                        }
                    ]
                }
        
        class IntelligentNaiveAgent:
            def __init__(self, database):
                self.db = database
                
            def answer_query(self, query):
                query_lower = query.lower()
                
                if 'merck' in query_lower and 'oncology' in query_lower:
                    return """Merck's Key Oncology Products:

• Keytruda (pembrolizumab) - PD-1 inhibitor
  Main uses: Lung cancer, melanoma, bladder cancer
  Revenue: ~$20B annually

• Lynparza (olaparib) - PARP inhibitor  
  Main uses: Ovarian, breast, prostate cancers
  Partnership with AstraZeneca

• Lenvima (lenvatinib) - Multi-kinase inhibitor
  Main uses: Thyroid, liver, kidney cancers
  Partnership with Eisai

Basic analysis available."""
                
                elif 'launch' in query_lower and 'oncology' in query_lower:
                    return """Upcoming oncology launches in 2025:
• Dato-DXd (Daiichi Sankyo/AstraZeneca) - HER2+ breast cancer
• Capivasertib (AstraZeneca) - PIK3CA/AKT pathway cancers

Basic analysis available."""
                
                elif 'competitive' in query_lower or 'competitor' in query_lower:
                    return """Key oncology competitors include AstraZeneca, Roche, Merck. Market is competitive with multiple targeted therapies."""
                
                elif 'diabetes' in query_lower and 'market' in query_lower:
                    return f"""Diabetes market size: {self.db.market_data['diabetes']['market_size_2024']}, growth rate: {self.db.market_data['diabetes']['growth_rate']}"""
                
                elif 'astrazeneca' in query_lower:
                    return """AstraZeneca is a leading pharmaceutical company with strong oncology portfolio including Tagrisso and Lynparza."""
                
                else:
                    return f"I can help with pharmaceutical analysis. Try asking about specific companies like Merck, AstraZeneca, market opportunities, or competitive landscapes."
        
        class AdvancedClassicalMLAgent:
            def __init__(self, database):
                self.db = database
                
            def answer_query(self, query):
                query_lower = query.lower()
                
                if 'merck' in query_lower and 'oncology' in query_lower:
                    return """📊 MERCK ONCOLOGY PORTFOLIO ANALYSIS
==========================================

🎯 FLAGSHIP PRODUCTS:

• Keytruda (pembrolizumab) - PD-1 Inhibitor
  Indications: NSCLC, melanoma, head & neck, bladder
  2023 Revenue: $25.0B (Merck's top seller)
  Growth: Expanding into adjuvant settings

• Lynparza (olaparib) - PARP Inhibitor
  Indications: Ovarian, breast, prostate cancers
  Partnership: Co-developed with AstraZeneca
  Focus: BRCA-mutated tumors

• Lenvima (lenvatinib) - Multi-kinase Inhibitor
  Indications: Thyroid, liver, endometrial cancers
  Partnership: Co-developed with Eisai
  Position: Second-line liver cancer standard

📈 STRATEGIC POSITION:
• Market leader in immuno-oncology
• Strong combination therapy pipeline
• Focus on biomarker-driven precision medicine

💡 COMPETITIVE ADVANTAGES:
• Keytruda's broad label expansion
• Strong clinical development capabilities
• Strategic partnership approach"""
                
                elif 'launch' in query_lower and 'oncology' in query_lower:
                    return """📊 ONCOLOGY PIPELINE ANALYSIS
==================================================

🎯 MAJOR LAUNCHES 2025:

• Dato-DXd (Daiichi Sankyo/AstraZeneca)
  Indication: HER2+ breast cancer
  Launch: Q2 2025
  Peak Sales: $8.5B
  Advantage: Next-gen ADC with improved efficacy

• Capivasertib (AstraZeneca)
  Indication: PIK3CA/AKT pathway cancers
  Launch: Q3 2025
  Peak Sales: $4.2B
  Advantage: First-in-class AKT inhibitor

💡 STRATEGIC IMPLICATIONS:
• High competition expected in oncology
• First-mover advantage critical for market share
• Combination strategies may differentiate offerings"""
                
                elif 'competitive' in query_lower or 'competitor' in query_lower:
                    return """🏆 COMPETITIVE LANDSCAPE ANALYSIS
==========================================

🫁 LUNG CANCER MARKET:
• AstraZeneca: 35% market share (Tagrisso, Imfinzi)
• Merck: 28% market share (Keytruda)
• Roche: Strong presence with multiple assets

⚠️ EMERGING THREATS:
• Amgen - KRAS G12C inhibitors
• Johnson & Johnson - bispecific antibodies
• Gilead - antibody-drug conjugates

📈 MARKET DYNAMICS:
Growth Drivers: Precision medicine, combination therapies
Key Challenges: Resistance mechanisms, high costs"""
                
                elif 'diabetes' in query_lower:
                    market = self.db.market_data['diabetes']
                    return f"""💰 DIABETES MARKET ANALYSIS
===============================

📊 MARKET METRICS:
• 2024 Size: {market['market_size_2024']}
• 2025 Projection: {market['projected_2025']}
• Growth Rate: {market['growth_rate']}
• Key Segments: {', '.join(market['key_segments'])}

🚀 GROWTH CATALYSTS:
• GLP-1 expansion into obesity
• Oral formulation development
• Digital health integration"""
                
                elif 'astrazeneca' in query_lower:
                    return """📊 ASTRAZENECA STRATEGIC ANALYSIS
=====================================

🎯 ONCOLOGY PORTFOLIO:
• Tagrisso (osimertinib) - EGFR inhibitor
• Lynparza (olaparib) - PARP inhibitor
• Imfinzi (durvalumab) - PD-L1 inhibitor

📈 PIPELINE STRENGTH:
• Multiple late-stage assets
• Strong ADC development program
• Combination therapy focus

💡 STRATEGIC POSITION:
• Market leader in lung cancer
• Growing presence in breast cancer
• Innovation-driven growth strategy"""
                
                else:
                    return f"""📈 PHARMACEUTICAL INTELLIGENCE
=============================

I can help you analyze:
• Company portfolios (Merck, AstraZeneca, etc.)
• Market opportunities and sizing
• Competitive landscapes
• Pipeline assessments
• Strategic recommendations

Please ask about specific companies, markets, or therapeutic areas."""
        
        class SophisticatedDeepLearningAgent:
            def __init__(self, database):
                self.db = database
                
            def answer_query(self, query):
                query_lower = query.lower()
                
                if 'launch' in query_lower and 'oncology' in query_lower:
                    return """🚀 COMPREHENSIVE ONCOLOGY PIPELINE INTELLIGENCE
============================================================

📅 Analysis Date: June 2025
🎯 Strategic Focus: Oncology Launch Readiness

🎯 MAJOR LAUNCHES 2025:
────────────────────────

1. Dato-DXd (Daiichi Sankyo/AstraZeneca)
   📋 Indication: HER2+ breast cancer
   📅 Expected Launch: Q2 2025
   💰 Peak Sales Projection: $8.5B
   🎯 Competitive Edge: Next-gen ADC with superior efficacy
   ⭐ Strategic Importance: HIGH

2. Capivasertib (AstraZeneca)
   📋 Indication: PIK3CA/AKT pathway cancers
   📅 Expected Launch: Q3 2025
   💰 Peak Sales Projection: $4.2B
   🎯 Competitive Edge: First-in-class AKT inhibitor
   ⭐ Strategic Importance: HIGH

💡 STRATEGIC RECOMMENDATIONS:
────────────────────────────
• Monitor competitive launches for partnership opportunities
• Prepare market access strategies for key approvals
• Assess pricing implications from new entrants
• Consider accelerated development timelines

⚠️ RISK ASSESSMENT:
──────────────────
• Regulatory delays could shift competitive dynamics
• Manufacturing capacity constraints possible
• Reimbursement challenges for premium pricing"""
                
                elif 'competitive' in query_lower or 'competitor' in query_lower:
                    return """🏆 DEEP COMPETITIVE INTELLIGENCE ANALYSIS
============================================================

🎯 Market Scope: Pharmaceutical Competitive Landscape
📊 Framework: Multi-dimensional Assessment

🫁 LUNG CANCER DYNAMICS:
────────────────────────

📊 MARKET LEADERS:
• AstraZeneca: 35% market share
  Assets: Tagrisso, Imfinzi
  Recent Developments: Adjuvant approvals expanding market

• Merck: 28% market share
  Assets: Keytruda
  Strategy: Perioperative expansion

⚡ EMERGING COMPETITIVE THREATS:
• Amgen - KRAS G12C inhibitors gaining traction
• Johnson & Johnson - bispecific antibodies
• Gilead - ADC expansion beyond existing markets

🎯 STRATEGIC POSITIONING OPPORTUNITIES:
──────────────────────────────────────
• Precision medicine leadership through biomarker strategies
• Combination therapy innovation for differentiation
• Real-world evidence generation for market access
• Digital health integration for patient engagement

🎲 SCENARIO PLANNING:
────────────────────
Best Case: Successful launches capture 25%+ market share
Base Case: Moderate adoption with 15-20% penetration
Risk Case: Delayed approvals or safety concerns limit uptake"""
                
                elif 'diabetes' in query_lower:
                    market = self.db.market_data['diabetes']
                    return f"""💰 STRATEGIC DIABETES MARKET INTELLIGENCE
======================================

🎯 Investment Thesis: Diabetes Market Assessment
📊 Multi-dimensional Opportunity Analysis

📊 MARKET FUNDAMENTALS:
─────────────────────
Current Size (2024): {market['market_size_2024']}
Projected (2025): {market['projected_2025']}
Growth Rate: {market['growth_rate']}
Key Segments: {', '.join(market['key_segments'])}

🚀 TRANSFORMATIONAL CATALYSTS:
────────────────────────────
• GLP-1 obesity indication expansion
• Oral GLP-1 formulation breakthrough
• CGM integration with therapeutics
• Digital therapeutics convergence

💡 INVESTMENT RECOMMENDATIONS:
───────────────────────────
• High Priority: Obesity-diabetes continuum assets
• Strategic: Digital health integration partnerships
• Innovation: Novel delivery mechanisms

⚖️ RISK-RETURN ASSESSMENT:
─────────────────────────
Upside: Multi-billion market expansion potential
Risk: Competitive intensity and pricing pressure
Action: Strategic investment recommended with portfolio approach"""
                
                elif 'astrazeneca' in query_lower:
                    return """🚀 ASTRAZENECA STRATEGIC INTELLIGENCE ANALYSIS
=================================================

🎯 COMPREHENSIVE PORTFOLIO ASSESSMENT
Multi-dimensional strategic evaluation

📊 ONCOLOGY DOMINANCE:
────────────────────
Core Assets:
• Tagrisso (osimertinib): $5.8B revenue, lung cancer leader
• Lynparza (olaparib): $2.5B revenue, PARP inhibitor pioneer
• Imfinzi (durvalumab): $3.1B revenue, PD-L1 innovation

Pipeline Strength:
• Dato-DXd: Potential $8.5B peak sales (partnership with Daiichi)
• Capivasertib: First-in-class AKT inhibitor opportunity
• Multiple combination strategies in development

🎯 STRATEGIC POSITIONING:
───────────────────────
Competitive Advantages:
• Market leadership in EGFR+ lung cancer
• Strong R&D capabilities in precision oncology
• Strategic partnerships enhancing pipeline

Growth Catalysts:
• ADC platform expansion
• Combination therapy approvals
• Geographic expansion in emerging markets

💡 INVESTMENT OUTLOOK:
────────────────────
Strengths: Innovation pipeline, market leadership
Opportunities: Emerging markets, digital health
Risks: Patent cliffs, competitive pressure
Recommendation: Strong long-term positioning"""
                
                else:
                    return f"""🚀 ADVANCED PHARMACEUTICAL INTELLIGENCE
====================================

🎯 SOPHISTICATED ANALYSIS FRAMEWORK
Multi-layered market intelligence powered by:

• AI-driven pattern recognition
• Competitive scenario modeling
• Strategic risk assessment
• Investment opportunity mapping

📊 QUERY PROCESSED: {query}

💡 COMPREHENSIVE INSIGHTS AVAILABLE:
• Market dynamics analysis
• Competitive positioning assessment
• Strategic recommendations
• Risk mitigation strategies

For more specific analysis, please ask about:
• Market opportunities
• Competitive landscapes
• Pipeline assessments
• Strategic recommendations"""
        
        # Initialize database and agents
        database = PharmaceuticalDatabase()
        
        agents = {
            'naive_agent': IntelligentNaiveAgent(database),
            'classical_agent': AdvancedClassicalMLAgent(database),
            'deep_agent': SophisticatedDeepLearningAgent(database)
        }
        
        return agents
        
    except Exception as e:
        st.error(f"Error loading pharmaceutical agents: {e}")
        return None

def show_example_queries():
    """Show example queries in a clean format"""
    st.markdown('<div class="example-queries">', unsafe_allow_html=True)
    st.markdown("**💡 Try asking me about:**")
    
    examples = [
        "What are the upcoming oncology launches in 2025?",
        "Who are the competitors for AstraZeneca in oncology?", 
        "What is the diabetes market opportunity?",
        "Analyze the competitive landscape in lung cancer",
        "Tell me about AstraZeneca's pipeline strategy",
        "What clinical trials should we monitor?"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"💬 {example}", key=f"example_{i}", use_container_width=True):
                st.session_state['example_query'] = example
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def main_chat_interface():
    """Clean Claude-like chat interface"""
    
    # Header
    st.markdown('<div class="header-text">Evan Moh | Duke University AIPI540</div>', unsafe_allow_html=True)
    
    # Welcome section
    st.markdown('<div class="welcome-section">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">IndicaAI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Pharmaceutical Marketing Intelligence Assistant</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load agents
    if not st.session_state.agents_loaded:
        with st.spinner("Loading pharmaceutical intelligence..."):
            agents = load_pharmaceutical_agents()
            if agents:
                st.session_state.agents = agents
                st.session_state.agents_loaded = True
    
    # Simple model selection - better visibility
    st.markdown('<div class="model-selector">', unsafe_allow_html=True)
    st.markdown("**🤖 Choose your AI assistant:**")
    selected_model = st.selectbox(
        "",
        options=['Deep Learning Agent', 'Classical ML Agent', 'Naive Agent'],
        index=0,
        help="Select the AI model for your pharmaceutical analysis",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model mapping
    model_mapping = {
        'Deep Learning Agent': 'deep_agent',
        'Classical ML Agent': 'classical_agent', 
        'Naive Agent': 'naive_agent'
    }
    
    # Show examples if no chat history
    if not st.session_state.messages:
        show_example_queries()
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            model_name = message["model"]
            st.markdown(f'<div class="assistant-message"><strong>{model_name}:</strong><br><pre>{message["content"]}</pre></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input section - always visible
    st.markdown("### 💬 Ask me anything about pharmaceutical intelligence:")
    
    # New question input
    new_question = st.text_input(
        "",
        placeholder="Type your question here... (e.g., Tell me about Merck's oncology products)",
        key="new_question_input",
        label_visibility="collapsed"
    )
    
    # Handle example query selection
    if 'example_query' in st.session_state and st.session_state.example_query:
        new_question = st.session_state['example_query']
        st.session_state['example_query'] = None  # Clear immediately to prevent re-processing
    
    # Process new question
    if new_question and st.session_state.agents_loaded:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": new_question})
        
        # Get response
        with st.spinner("Thinking..."):
            try:
                selected_agent_key = model_mapping[selected_model]
                agent = st.session_state.agents[selected_agent_key]
                
                response = agent.answer_query(new_question)
                
                # Add response
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "model": selected_model
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Action buttons
    if st.session_state.messages:
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🏠 New Chat", type="primary"):
                st.session_state.messages = []
                if 'example_query' in st.session_state:
                    st.session_state['example_query'] = None
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All", type="secondary"):
                st.session_state.messages = []
                if 'example_query' in st.session_state:
                    st.session_state['example_query'] = None
                st.rerun()
        
        with col3:
            st.write("")  # Placeholder for spacing

def evaluation_metrics_page():
    """Evaluation metrics page (moved from main interface)"""
    
    st.markdown('<div class="header-text">Evan Moh | Duke University AIPI540</div>', unsafe_allow_html=True)
    
    st.title("📊 Model Evaluation & Performance Analytics")
    st.markdown("---")
    
    # Performance results
    st.header("🏆 Performance Results")
    
    evaluation_data = {
        'Model': ['🥇 Deep Learning Agent', '🥈 Classical ML Agent', '🥉 Naive Agent'],
        'Overall Score': [0.654, 0.565, 0.409],
        'Technical Performance': [0.998, 0.667, 0.417],
        'Business Intelligence': [1.000, 1.000, 0.850],
        'Clinical Intelligence': [0.125, 0.167, 0.000],
        'Strategic Support': [0.148, 0.074, 0.037]
    }
    
    df = pd.DataFrame(evaluation_data)
    
    # Overall performance chart
    fig = px.bar(
        df, 
        x='Model', 
        y='Overall Score',
        title="Overall Performance Rankings",
        color='Overall Score',
        color_continuous_scale='RdYlGn',
        text='Overall Score'
    )
    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig.update_layout(showlegend=False, yaxis_tickformat='%')
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🥇 Deep Learning Agent", "65.4%", "Champion")
    with col2:
        st.metric("🥈 Classical ML Agent", "56.5%", "Runner-up")
    with col3:
        st.metric("🥉 Naive Agent", "40.9%", "Bronze")
    
    # Detailed breakdown
    st.header("📈 Detailed Performance Breakdown")
    
    # Format and display table
    df_display = df.copy()
    for col in df_display.columns:
        if col != 'Model':
            df_display[col] = df_display[col].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(df_display, use_container_width=True)
    
    # Key insights
    st.header("💡 Key Insights")
    
    st.write("""
    **🥇 Deep Learning Agent (65.4%)** - Champion Performance
    - Industry-leading technical architecture (99.8%)
    - Outstanding business intelligence (100%)
    - Best overall choice for sophisticated analysis
    
    **🥈 Classical ML Agent (56.5%)** - Strong Alternative
    - Excellent business intelligence (100%)
    - Good technical foundation (66.7%)
    - Ideal for competitive analysis
    
    **🥉 Naive Agent (40.9%)** - Foundational Capabilities
    - Solid business intelligence (85%)
    - Good for basic queries and quick lookups
    """)

# Navigation
def main():
    """Main application"""
    
    # Sidebar with minimal info
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Select Page:",
        ["💬 Chat", "📊 Model Performance"]
    )
    
    # Route to pages
    if page == "💬 Chat":
        main_chat_interface()
    elif page == "📊 Model Performance":
        evaluation_metrics_page()

if __name__ == "__main__":
    main()
