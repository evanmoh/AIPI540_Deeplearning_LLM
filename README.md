# IndicaAI - Pharmaceutical Marketing Intelligence Platform

**Project for Evan Moh | Duke University AIPI540**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 🎯 Project Overview

IndicaAI is an advanced pharmaceutical marketing intelligence platform that leverages artificial intelligence to provide strategic insights for oncology markets. The platform compares three different AI model architectures to demonstrate the evolution from basic to sophisticated pharmaceutical intelligence.

## 🚀 Features

### Interactive Chat Interface
- **Real-time Query Processing**: Ask questions about pharmaceutical oncology markets
- **Multi-Model Comparison**: Choose between three AI agents with different capabilities
- **Professional UI**: Clean, intuitive interface designed for business stakeholders

### Pharmaceutical Intelligence Capabilities
- **Pipeline Analysis**: Upcoming drug launches and market opportunities
- **Competitive Intelligence**: Market share analysis and competitive positioning
- **Market Opportunity Assessment**: Market sizing and growth forecasting
- **Clinical Intelligence**: Trial data and regulatory insights
- **Strategic Recommendations**: Investment and positioning guidance

### Comprehensive Model Evaluation
- **Performance Metrics**: Technical, business, clinical, and strategic evaluation
- **Comparative Analysis**: Side-by-side model performance comparison
- **Interactive Visualizations**: Charts and graphs showing model capabilities

## 🤖 AI Model Architecture

### 1. Naive Baseline Agent
- **Approach**: Simple keyword matching and template responses
- **Capabilities**: Basic pharmaceutical query processing
- **Use Case**: Quick lookups and simple information retrieval
- **Performance**: 29.5% overall score

### 2. Classical Machine Learning Agent
- **Approach**: TF-IDF vectorization + Random Forest classification
- **Capabilities**: Enhanced market intelligence with structured analysis
- **Use Case**: Detailed market analysis and competitive intelligence
- **Performance**: 40.7% overall score

### 3. Deep Learning Agent
- **Approach**: LSTM + Attention mechanisms with pharmaceutical vocabulary
- **Capabilities**: Sophisticated analysis with executive-level insights
- **Use Case**: Strategic planning and comprehensive market intelligence
- **Performance**: 45.7% overall score

## 📊 Evaluation Framework

The platform includes a comprehensive evaluation framework measuring:

- **Technical Performance** (25% weight)
  - Classification accuracy
  - Response relevance
  - Entity extraction
  - Response time
  - Consistency

- **Business Intelligence** (35% weight)
  - Market data accuracy
  - Competitive intelligence quality
  - Pipeline intelligence
  - Strategic insight depth

- **Clinical Intelligence** (20% weight)
  - Clinical terminology usage
  - Safety information inclusion

- **Strategic Decision Support** (20% weight)
  - Strategic insight quality
  - Decision support capability
  - Risk assessment

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: PyTorch, scikit-learn, transformers
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly, matplotlib, seaborn
- **Natural Language Processing**: Custom pharmaceutical vocabulary and tokenization

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/indicaai-pharmaceutical-intelligence.git
cd indicaai-pharmaceutical-intelligence
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

### 4. Access the Platform
Open your browser and navigate to `http://localhost:8501`

## 🌐 Live Deployment

The application is deployed on Streamlit Cloud and accessible at: [Your Deployment URL]

## 📁 Project Structure

```
indicaai-pharmaceutical-intelligence/
├── app.py                          # Streamlit web application
├── pharmaceutical_intelligence.py  # Core AI models and database
├── evaluation_framework.py         # Model evaluation system
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .streamlit/
    └── config.toml                 # Streamlit configuration
```

## 🧪 Sample Queries

Try these example queries in the chat interface:

- "What are the upcoming oncology launches in 2025?"
- "Analyze the competitive landscape in lung cancer"
- "What is the market opportunity for diabetes drugs?"
- "Which clinical trials should we monitor in 2025?"
- "AstraZeneca pipeline strategy analysis"
- "What are the side effects of osimertinib?"

## 📈 Performance Results

| Model | Overall Score | Technical | Business | Clinical | Strategic |
|-------|---------------|-----------|----------|----------|-----------|
| Deep Learning | 45.7% | 91.7% | 52.5% | 0.0% | 22.2% |
| Classical ML | 40.7% | 90.0% | 50.0% | 0.0% | 3.7% |
| Naive Baseline | 29.5% | 81.2% | 26.4% | 0.0% | 0.0% |

## 🎓 Academic Context

This project was developed as part of Duke University's AIPI540 course, demonstrating:

- **AI Model Comparison**: Naive vs Classical ML vs Deep Learning approaches
- **Domain-Specific Applications**: Pharmaceutical market intelligence
- **Evaluation Methodologies**: Comprehensive multi-dimensional assessment
- **Practical Deployment**: Production-ready web application

## 🔬 Research & Data Sources

The platform utilizes simulated pharmaceutical data representing:
- Market sizing and growth projections
- Competitive landscape analysis
- Drug pipeline and launch timelines
- Clinical trial information
- Strategic market insights

*Note: This is an educational project using simulated data for demonstration purposes.*

## 👥 Author

**Evan Moh**  
Duke University - AIPI540  
[LinkedIn](https://linkedin.com/in/evanmoh) | [GitHub](https://github.com/evanmoh)

## 📄 License

This project is created for educational purposes as part of Duke University coursework.

## 🙏 Acknowledgments

- Duke University AIPI540 Course
- Pharmaceutical industry data and insights
- Open source AI/ML community

---

*Last Updated: june 2025*
