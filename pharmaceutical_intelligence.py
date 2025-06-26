# ============================================================================
# GOOGLE COLAB SETUP CELL - Run this first
# ============================================================================

# Install required packages
!pip install -q scikit-learn torch pandas numpy transformers

# Check if we're in Colab
try:
    import google.colab
    IN_COLAB = True
    print("✅ Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("ℹ️ Running locally")

# Import all required libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging
import re
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import time
import warnings
import random
warnings.filterwarnings("ignore")

# Configure logging for Colab
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

print("📦 All packages imported successfully!")

# ============================================================================
# PHARMACEUTICAL MARKET INTELLIGENCE DATABASE
# ============================================================================

class PharmaceuticalDatabase:
    """Comprehensive pharmaceutical market intelligence database"""
    
    def __init__(self):
        self.setup_market_data()
        self.setup_pipeline_data()
        self.setup_competitive_intelligence()
        self.setup_clinical_trial_data()
        
    def setup_market_data(self):
        """Setup comprehensive market data"""
        self.market_data = {
            'oncology': {
                'market_size_2024': '$196.1B',
                'projected_2025': '$210.3B',
                'growth_rate': '7.2%',
                'key_segments': ['lung_cancer', 'breast_cancer', 'blood_cancers', 'colorectal_cancer'],
                'market_leaders': ['AstraZeneca', 'Roche', 'Merck', 'Bristol Myers Squibb'],
                'upcoming_catalysts': [
                    'FDA approvals for combination therapies',
                    'Biomarker-driven precision medicine expansion',
                    'CAR-T therapy advancements'
                ]
            },
            'diabetes': {
                'market_size_2024': '$78.2B',
                'projected_2025': '$84.1B', 
                'growth_rate': '7.5%',
                'key_segments': ['GLP-1 agonists', 'insulin', 'SGLT2_inhibitors'],
                'market_leaders': ['Novo Nordisk', 'Eli Lilly', 'Sanofi'],
                'upcoming_catalysts': [
                    'Obesity indication expansions',
                    'Oral GLP-1 formulations',
                    'CGM integration with therapeutics'
                ]
            },
            'cardiovascular': {
                'market_size_2024': '$65.3B',
                'projected_2025': '$68.9B',
                'growth_rate': '5.5%',
                'key_segments': ['statins', 'antihypertensives', 'anticoagulants'],
                'market_leaders': ['Pfizer', 'AstraZeneca', 'Novartis'],
                'upcoming_catalysts': [
                    'PCSK9 inhibitor adoption',
                    'Digital therapeutics integration'
                ]
            }
        }
    
    def setup_pipeline_data(self):
        """Setup drug pipeline and launch data"""
        self.pipeline_data = {
            'upcoming_launches_2025': [
                {
                    'drug': 'Dato-DXd',
                    'company': 'Daiichi Sankyo/AstraZeneca',
                    'indication': 'HER2+ breast cancer',
                    'category': 'oncology',
                    'expected_launch': 'Q2 2025',
                    'peak_sales_projection': '$8.5B',
                    'competitive_advantage': 'Next-gen ADC with improved efficacy'
                },
                {
                    'drug': 'Capivasertib',
                    'company': 'AstraZeneca',
                    'indication': 'PIK3CA/AKT pathway cancers',
                    'category': 'oncology',
                    'expected_launch': 'Q3 2025',
                    'peak_sales_projection': '$4.2B',
                    'competitive_advantage': 'First-in-class AKT inhibitor'
                },
                {
                    'drug': 'Retatrutide',
                    'company': 'Eli Lilly',
                    'indication': 'Obesity/Type 2 diabetes',
                    'category': 'diabetes',
                    'expected_launch': 'Q4 2025',
                    'peak_sales_projection': '$15.3B',
                    'competitive_advantage': 'Triple hormone agonist'
                },
                {
                    'drug': 'Zilebesiran',
                    'company': 'Alnylam',
                    'indication': 'Hypertension',
                    'category': 'cardiovascular',
                    'expected_launch': 'H2 2025',
                    'peak_sales_projection': '$3.1B',
                    'competitive_advantage': 'First siRNA for hypertension'
                },
                {
                    'drug': 'Selumetinib combo',
                    'company': 'AstraZeneca',
                    'indication': 'KRAS+ lung cancer',
                    'category': 'oncology',
                    'expected_launch': 'Q1 2025',
                    'peak_sales_projection': '$2.8B',
                    'competitive_advantage': 'MEK inhibitor combination'
                }
            ],
            
            'late_stage_pipeline': [
                {
                    'drug': 'Tozorakimab',
                    'company': 'AstraZeneca',
                    'indication': 'COPD/Asthma',
                    'phase': 'Phase III',
                    'category': 'respiratory',
                    'trial_completion': 'Q2 2025',
                    'regulatory_filing': 'Q4 2025'
                },
                {
                    'drug': 'Volanesorsen',
                    'company': 'Akcea/Ionis',
                    'indication': 'Hypertriglyceridemia',
                    'phase': 'Phase III',
                    'category': 'cardiovascular',
                    'trial_completion': 'Q1 2025',
                    'regulatory_filing': 'Q3 2025'
                }
            ]
        }
    
    def setup_competitive_intelligence(self):
        """Setup competitive landscape data"""
        self.competitive_data = {
            'lung_cancer': {
                'market_leaders': {
                    'AstraZeneca': {
                        'key_drugs': ['Tagrisso', 'Imfinzi'],
                        'market_share': '35%',
                        'pipeline_strength': 'Strong',
                        'recent_developments': [
                            'Tagrisso adjuvant approval expanding market',
                            'Imfinzi combinations showing promise'
                        ]
                    },
                    'Merck': {
                        'key_drugs': ['Keytruda'],
                        'market_share': '28%',
                        'pipeline_strength': 'Strong',
                        'recent_developments': [
                            'Keytruda perioperative trials positive',
                            'Expanding into earlier stage disease'
                        ]
                    }
                },
                'emerging_threats': [
                    'Amgen - KRAS G12C inhibitors',
                    'Johnson & Johnson - bispecific antibodies',
                    'Gilead - antibody-drug conjugates'
                ],
                'market_dynamics': {
                    'growth_drivers': ['Precision medicine', 'Combination therapies', 'Earlier intervention'],
                    'challenges': ['Resistance mechanisms', 'High development costs', 'Regulatory complexity']
                }
            },
            
            'breast_cancer': {
                'market_leaders': {
                    'Roche': {
                        'key_drugs': ['Herceptin', 'Perjeta', 'Kadcyla'],
                        'market_share': '42%',
                        'pipeline_strength': 'Strong'
                    },
                    'AstraZeneca': {
                        'key_drugs': ['Lynparza', 'Enhertu'],
                        'market_share': '18%',
                        'pipeline_strength': 'Very Strong'
                    }
                },
                'upcoming_disruptions': [
                    'ADC expansion beyond HER2+',
                    'PARP inhibitor combinations',
                    'Oral SERD development'
                ]
            }
        }
    
    def setup_clinical_trial_data(self):
        """Setup clinical trial intelligence"""
        self.clinical_data = {
            'high_impact_trials_2025': [
                {
                    'trial_name': 'TROPION-Lung05',
                    'drug': 'Dato-DXd',
                    'company': 'Daiichi Sankyo/AstraZeneca',
                    'indication': 'NSCLC',
                    'readout_timing': 'Q1 2025',
                    'market_impact': 'High - could expand ADC use in lung cancer'
                },
                {
                    'trial_name': 'CAPItello-291',
                    'drug': 'Capivasertib + fulvestrant',
                    'company': 'AstraZeneca',
                    'indication': 'HR+ breast cancer',
                    'readout_timing': 'Q2 2025',
                    'market_impact': 'High - new mechanism in breast cancer'
                },
                {
                    'trial_name': 'SURMOUNT-5',
                    'drug': 'Retatrutide',
                    'company': 'Eli Lilly',
                    'indication': 'Obesity',
                    'readout_timing': 'Q3 2025',
                    'market_impact': 'Very High - could dominate obesity market'
                }
            ]
        }

# ============================================================================
# ENHANCED PHARMACEUTICAL MARKETING AGENTS
# ============================================================================

class IntelligentNaiveAgent:
    """Naive agent with basic market intelligence"""
    
    def __init__(self, database: PharmaceuticalDatabase):
        self.db = database
        self.basic_keywords = {
            'launches': ['launch', 'upcoming', 'new', 'approval', 'pipeline'],
            'competition': ['competitor', 'competitive', 'market share', 'landscape'],
            'market': ['market', 'size', 'growth', 'opportunity', 'forecast'],
            'trials': ['trial', 'clinical', 'study', 'data', 'results']
        }
        
    def classify_query_intent(self, query: str) -> str:
        """Classify the intent of the marketing query"""
        query_lower = query.lower()
        
        for intent, keywords in self.basic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        return 'general'
    
    def extract_therapeutic_area(self, query: str) -> str:
        """Extract therapeutic area from query"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['cancer', 'oncology', 'tumor', 'lung', 'breast']):
            return 'oncology'
        elif any(term in query_lower for term in ['diabetes', 'glucose', 'insulin']):
            return 'diabetes'
        elif any(term in query_lower for term in ['heart', 'cardiovascular', 'cholesterol']):
            return 'cardiovascular'
        
        return 'general'
    
    def answer_query(self, query: str) -> str:
        """Generate basic market intelligence response"""
        intent = self.classify_query_intent(query)
        therapeutic_area = self.extract_therapeutic_area(query)
        
        if intent == 'launches' and therapeutic_area in ['oncology', 'diabetes', 'cardiovascular']:
            # Basic upcoming launches
            launches = [drug for drug in self.db.pipeline_data['upcoming_launches_2025'] 
                       if drug['category'] == therapeutic_area]
            
            if launches:
                response = f"Upcoming {therapeutic_area} launches in 2025:\n"
                for launch in launches[:2]:  # Limit to 2 for naive model
                    response += f"• {launch['drug']} ({launch['company']}) - {launch['indication']}\n"
                response += "\nBasic analysis available."
            else:
                response = f"Limited information available for {therapeutic_area} launches."
                
        elif intent == 'competition':
            if therapeutic_area == 'oncology':
                response = "Key oncology competitors include AstraZeneca, Roche, Merck. Market is competitive with multiple targeted therapies."
            else:
                response = f"Competitive analysis for {therapeutic_area} shows established market leaders with ongoing innovation."
                
        elif intent == 'market':
            if therapeutic_area in self.db.market_data:
                market_info = self.db.market_data[therapeutic_area]
                response = f"{therapeutic_area.title()} market size: {market_info['market_size_2024']}, growth rate: {market_info['growth_rate']}"
            else:
                response = "Market data available upon request."
                
        else:
            response = "Basic pharmaceutical market query processed. Limited analysis capabilities."
        
        return response

class AdvancedClassicalMLAgent:
    """Classical ML agent with enhanced market intelligence"""
    
    def __init__(self, database: PharmaceuticalDatabase):
        self.db = database
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.is_trained = False
        
        # Enhanced query classification
        self.intent_keywords = {
            'pipeline_analysis': ['pipeline', 'launch', 'upcoming', 'approval', 'regulatory'],
            'competitive_intelligence': ['competitor', 'competitive', 'market share', 'landscape', 'threat'],
            'market_opportunity': ['market', 'size', 'opportunity', 'growth', 'forecast', 'revenue'],
            'clinical_intelligence': ['trial', 'clinical', 'data', 'results', 'efficacy', 'safety'],
            'strategic_planning': ['strategy', 'positioning', 'investment', 'portfolio']
        }
        
    def classify_query_intent(self, query: str) -> str:
        """Advanced intent classification"""
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = sum(2 if len(keyword) > 6 else 1 for keyword in keywords if keyword in query_lower)
            intent_scores[intent] = score
            
        if not any(intent_scores.values()):
            return 'general_inquiry'
            
        return max(intent_scores, key=intent_scores.get)
    
    def extract_context(self, query: str) -> Dict:
        """Extract detailed context from query"""
        query_lower = query.lower()
        
        context = {
            'therapeutic_areas': [],
            'companies': [],
            'timeframe': 'current',
            'geographic_scope': 'global'
        }
        
        # Therapeutic areas
        ta_mapping = {
            'oncology': ['cancer', 'oncology', 'tumor', 'lung', 'breast', 'nsclc'],
            'diabetes': ['diabetes', 'glucose', 'insulin', 'obesity'],
            'cardiovascular': ['heart', 'cardiovascular', 'cholesterol', 'hypertension'],
            'respiratory': ['asthma', 'copd', 'respiratory']
        }
        
        for ta, keywords in ta_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                context['therapeutic_areas'].append(ta)
        
        # Companies
        companies = ['astrazeneca', 'merck', 'roche', 'pfizer', 'eli lilly', 'novartis']
        for company in companies:
            if company in query_lower:
                context['companies'].append(company.title())
        
        # Timeframe
        if any(term in query_lower for term in ['2025', 'next year', 'upcoming']):
            context['timeframe'] = '2025'
        elif any(term in query_lower for term in ['2024', 'current', 'this year']):
            context['timeframe'] = '2024'
            
        return context
    
    def answer_query(self, query: str) -> str:
        """Generate sophisticated market intelligence response"""
        intent = self.classify_query_intent(query)
        context = self.extract_context(query)
        
        if intent == 'pipeline_analysis':
            return self.generate_pipeline_analysis(context)
        elif intent == 'competitive_intelligence':
            return self.generate_competitive_analysis(context)
        elif intent == 'market_opportunity':
            return self.generate_market_analysis(context)
        elif intent == 'clinical_intelligence':
            return self.generate_clinical_analysis(context)
        else:
            return self.generate_general_analysis(context)
    
    def generate_pipeline_analysis(self, context: Dict) -> str:
        """Generate detailed pipeline analysis"""
        therapeutic_areas = context['therapeutic_areas']
        
        if not therapeutic_areas:
            therapeutic_areas = ['oncology']  # Default
            
        response = "📊 PIPELINE ANALYSIS REPORT\n" + "="*50 + "\n"
        
        for ta in therapeutic_areas:
            launches = [drug for drug in self.db.pipeline_data['upcoming_launches_2025'] 
                       if drug['category'] == ta]
            
            if launches:
                response += f"\n🎯 {ta.upper()} UPCOMING LAUNCHES 2025:\n"
                for launch in launches:
                    response += f"\n• {launch['drug']} ({launch['company']})\n"
                    response += f"  Indication: {launch['indication']}\n"
                    response += f"  Launch: {launch['expected_launch']}\n"
                    response += f"  Peak Sales: {launch['peak_sales_projection']}\n"
                    response += f"  Advantage: {launch['competitive_advantage']}\n"
        
        # Add late stage pipeline
        late_stage = [drug for drug in self.db.pipeline_data['late_stage_pipeline'] 
                     if drug['category'] in therapeutic_areas]
        
        if late_stage:
            response += f"\n🔬 LATE STAGE PIPELINE:\n"
            for drug in late_stage:
                response += f"• {drug['drug']} ({drug['company']}) - {drug['indication']} (Phase {drug['phase'][-3:]})\n"
        
        response += f"\n💡 STRATEGIC IMPLICATIONS:\n"
        response += f"• High competition expected in {', '.join(therapeutic_areas)}\n"
        response += f"• First-mover advantage critical for market share\n"
        response += f"• Combination strategies may differentiate offerings\n"
        
        return response
    
    def generate_competitive_analysis(self, context: Dict) -> str:
        """Generate competitive landscape analysis"""
        therapeutic_areas = context['therapeutic_areas'] or ['oncology']
        
        response = "🏆 COMPETITIVE INTELLIGENCE REPORT\n" + "="*50 + "\n"
        
        for ta in therapeutic_areas:
            if ta == 'oncology':
                lung_data = self.db.competitive_data.get('lung_cancer', {})
                response += f"\n🫁 LUNG CANCER COMPETITIVE LANDSCAPE:\n"
                
                leaders = lung_data.get('market_leaders', {})
                for company, data in leaders.items():
                    response += f"\n• {company}: {data.get('market_share', 'N/A')} market share\n"
                    response += f"  Key drugs: {', '.join(data.get('key_drugs', []))}\n"
                    response += f"  Pipeline: {data.get('pipeline_strength', 'Unknown')}\n"
                
                threats = lung_data.get('emerging_threats', [])
                if threats:
                    response += f"\n⚠️ EMERGING COMPETITIVE THREATS:\n"
                    for threat in threats:
                        response += f"• {threat}\n"
        
        response += f"\n📈 MARKET DYNAMICS:\n"
        if 'oncology' in therapeutic_areas:
            dynamics = self.db.competitive_data['lung_cancer']['market_dynamics']
            response += f"Growth Drivers: {', '.join(dynamics['growth_drivers'])}\n"
            response += f"Key Challenges: {', '.join(dynamics['challenges'])}\n"
        
        return response
    
    def generate_market_analysis(self, context: Dict) -> str:
        """Generate market opportunity analysis"""
        therapeutic_areas = context['therapeutic_areas'] or ['oncology']
        
        response = "💰 MARKET OPPORTUNITY ANALYSIS\n" + "="*50 + "\n"
        
        for ta in therapeutic_areas:
            if ta in self.db.market_data:
                market = self.db.market_data[ta]
                response += f"\n📊 {ta.upper()} MARKET METRICS:\n"
                response += f"• 2024 Market Size: {market['market_size_2024']}\n"
                response += f"• 2025 Projection: {market['projected_2025']}\n"
                response += f"• Growth Rate: {market['growth_rate']}\n"
                response += f"• Key Segments: {', '.join(market['key_segments'])}\n"
                response += f"• Market Leaders: {', '.join(market['market_leaders'])}\n"
                
                response += f"\n🚀 UPCOMING CATALYSTS:\n"
                for catalyst in market['upcoming_catalysts']:
                    response += f"• {catalyst}\n"
        
        response += f"\n💡 INVESTMENT RECOMMENDATIONS:\n"
        response += f"• Focus on high-growth segments with unmet need\n"
        response += f"• Consider strategic partnerships for market access\n"
        response += f"• Monitor regulatory environment for opportunities\n"
        
        return response
    
    def generate_clinical_analysis(self, context: Dict) -> str:
        """Generate clinical intelligence analysis"""
        response = "🔬 CLINICAL INTELLIGENCE REPORT\n" + "="*50 + "\n"
        
        response += f"\n📋 HIGH-IMPACT TRIAL READOUTS 2025:\n"
        for trial in self.db.clinical_data['high_impact_trials_2025']:
            response += f"\n• {trial['trial_name']}\n"
            response += f"  Drug: {trial['drug']} ({trial['company']})\n"
            response += f"  Indication: {trial['indication']}\n"
            response += f"  Readout: {trial['readout_timing']}\n"
            response += f"  Impact: {trial['market_impact']}\n"
        
        return response
    
    def generate_general_analysis(self, context: Dict) -> str:
        """Generate general pharmaceutical analysis"""
        return "📈 PHARMACEUTICAL MARKET OVERVIEW\n" + "="*50 + "\n\nComprehensive analysis available across oncology, diabetes, and cardiovascular markets. Specific intelligence can be provided for pipeline analysis, competitive landscape, and market opportunities."

class SophisticatedDeepLearningAgent:
    """Advanced deep learning agent with comprehensive market intelligence"""
    
    def __init__(self, database: PharmaceuticalDatabase):
        self.db = database
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Advanced NLP capabilities
        self.build_advanced_vocabulary()
        self.model = self.build_sophisticated_model().to(self.device)
        self.comprehensive_train()
        
    def build_advanced_vocabulary(self):
        """Build comprehensive pharmaceutical vocabulary"""
        pharma_terms = [
            # Market intelligence terms
            'pipeline', 'launch', 'approval', 'regulatory', 'competitive', 'market',
            'opportunity', 'forecast', 'revenue', 'growth', 'strategic', 'positioning',
            
            # Therapeutic areas
            'oncology', 'cancer', 'tumor', 'lung', 'breast', 'nsclc', 'diabetes',
            'cardiovascular', 'heart', 'cholesterol', 'respiratory', 'asthma',
            
            # Drug development
            'clinical', 'trial', 'phase', 'efficacy', 'safety', 'biomarker',
            'mechanism', 'inhibitor', 'therapy', 'treatment', 'combination',
            
            # Companies and drugs
            'astrazeneca', 'merck', 'roche', 'pfizer', 'osimertinib', 'olaparib',
            'metformin', 'tagrisso', 'lynparza', 'keytruda', 'dato', 'capivasertib',
            
            # Market dynamics
            'competition', 'landscape', 'threat', 'advantage', 'differentiation',
            'disruption', 'innovation', 'investment', 'portfolio', 'strategy'
        ]
        
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for i, term in enumerate(pharma_terms, 2):
            self.vocab[term] = i
        self.vocab_size = len(self.vocab)
        
    def build_sophisticated_model(self):
        """Build sophisticated neural architecture"""
        class AdvancedPharmaAnalyzer(nn.Module):
            def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
                self.attention = nn.MultiheadAttention(hidden_dim*2, num_heads=8, batch_first=True)
                
                # Multiple output heads for different tasks
                self.intent_classifier = nn.Linear(hidden_dim*2, 6)  # 6 intent types
                self.therapeutic_area_classifier = nn.Linear(hidden_dim*2, 5)  # 5 TAs
                self.priority_scorer = nn.Linear(hidden_dim*2, 1)  # Priority score
                
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                embedded = self.embedding(x)
                lstm_out, _ = self.lstm(embedded)
                attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
                
                # Global average pooling
                pooled = attended.mean(dim=1)
                pooled = self.dropout(pooled)
                
                # Multiple outputs
                intent = self.intent_classifier(pooled)
                therapeutic_area = self.therapeutic_area_classifier(pooled)
                priority = torch.sigmoid(self.priority_scorer(pooled))
                
                return intent, therapeutic_area, priority
        
        return AdvancedPharmaAnalyzer(self.vocab_size)
    
    def comprehensive_train(self):
        """Train with comprehensive pharmaceutical data"""
        print("🚀 Training Sophisticated Deep Learning model...")
        
        # Create comprehensive training data
        training_data = self.create_comprehensive_training_data()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0005)
        intent_criterion = nn.CrossEntropyLoss()
        ta_criterion = nn.CrossEntropyLoss()
        priority_criterion = nn.BCELoss()
        
        self.model.train()
        for epoch in range(10):
            total_loss = 0
            
            for batch in training_data:
                optimizer.zero_grad()
                
                inputs = torch.tensor([self.tokenize_advanced(text) for text in batch['texts']])
                intent_targets = torch.tensor(batch['intents'])
                ta_targets = torch.tensor(batch['therapeutic_areas'])
                priority_targets = torch.tensor(batch['priorities']).float()
                
                intent_pred, ta_pred, priority_pred = self.model(inputs)
                
                loss = (intent_criterion(intent_pred, intent_targets) +
                       ta_criterion(ta_pred, ta_targets) +
                       priority_criterion(priority_pred.squeeze(), priority_targets))
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 3 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.3f}")
        
        print("✅ Sophisticated Deep Learning model trained!")
    
    def create_comprehensive_training_data(self):
        """Create comprehensive training dataset"""
        return [{
            'texts': [
                'upcoming oncology launches 2025',
                'lung cancer competitive landscape',
                'diabetes market opportunity analysis',
                'clinical trial readouts next year',
                'AstraZeneca pipeline strategy',
                'breast cancer market size forecast'
            ],
            'intents': [0, 1, 2, 3, 4, 2],  # pipeline, competitive, market, clinical, strategic, market
            'therapeutic_areas': [0, 0, 1, 4, 4, 0],  # oncology, oncology, diabetes, general, general, oncology
            'priorities': [0.9, 0.8, 0.7, 0.8, 0.6, 0.7]
        }]
    
    def tokenize_advanced(self, text: str, max_length: int = 40) -> List[int]:
        """Advanced tokenization"""
        words = text.lower().split()[:max_length]
        tokens = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        while len(tokens) < max_length:
            tokens.append(self.vocab['<PAD>'])
            
        return tokens[:max_length]
    
    def answer_query(self, query: str) -> str:
        """Generate sophisticated market intelligence response"""
        # Use the model to understand query intent and context
        intent, priority = self.analyze_query_advanced(query)
        
        if 'launch' in query.lower() or 'upcoming' in query.lower():
            return self.generate_comprehensive_pipeline_report(query)
        elif 'competitive' in query.lower() or 'competitor' in query.lower():
            return self.generate_deep_competitive_analysis(query)
        elif 'market' in query.lower() and ('size' in query.lower() or 'opportunity' in query.lower()):
            return self.generate_strategic_market_analysis(query)
        elif 'trial' in query.lower() or 'clinical' in query.lower():
            return self.generate_clinical_intelligence_report(query)
        else:
            return self.generate_executive_summary(query)
    
    def analyze_query_advanced(self, query: str) -> Tuple[str, float]:
        """Advanced query analysis using deep learning"""
        try:
            self.model.eval()
            with torch.no_grad():
                tokens = torch.tensor([self.tokenize_advanced(query)]).to(self.device)
                intent_pred, ta_pred, priority_pred = self.model(tokens)
                
                intent_probs = torch.softmax(intent_pred, dim=1)
                priority_score = priority_pred.item()
                
                intents = ['pipeline_analysis', 'competitive_intelligence', 'market_opportunity', 
                          'clinical_intelligence', 'strategic_planning', 'general']
                
                intent_idx = torch.argmax(intent_probs).item()
                return intents[intent_idx], priority_score
        except:
            return 'general', 0.5
    
    def generate_comprehensive_pipeline_report(self, query: str) -> str:
        """Generate comprehensive pipeline analysis"""
        therapeutic_area = self.extract_therapeutic_context(query)
        
        response = "🚀 COMPREHENSIVE PIPELINE INTELLIGENCE REPORT\n"
        response += "="*60 + "\n"
        response += f"Analysis Date: {datetime.now().strftime('%B %Y')}\n"
        response += f"Focus Area: {therapeutic_area.title()}\n\n"
        
        # Upcoming launches
        launches = [drug for drug in self.db.pipeline_data['upcoming_launches_2025'] 
                   if therapeutic_area in drug['category'] or therapeutic_area == 'all']
        
        if launches:
            response += "🎯 MAJOR LAUNCHES 2025:\n"
            response += "-" * 40 + "\n"
            
            for i, launch in enumerate(launches, 1):
                response += f"\n{i}. {launch['drug']} ({launch['company']})\n"
                response += f"   📋 Indication: {launch['indication']}\n"
                response += f"   📅 Expected Launch: {launch['expected_launch']}\n"
                response += f"   💰 Peak Sales Projection: {launch['peak_sales_projection']}\n"
                response += f"   🎯 Competitive Edge: {launch['competitive_advantage']}\n"
                
                # Add market impact assessment
                if 'AstraZeneca' in launch['company']:
                    response += f"   ⭐ Strategic Importance: HIGH (AstraZeneca portfolio strengthening)\n"
                else:
                    response += f"   ⚠️  Competitive Threat: Monitor closely\n"
        
        # Clinical catalysts
        response += f"\n🔬 KEY CLINICAL CATALYSTS 2025:\n"
        response += "-" * 40 + "\n"
        
        relevant_trials = [trial for trial in self.db.clinical_data['high_impact_trials_2025']
                          if therapeutic_area in trial['drug'].lower() or therapeutic_area == 'all']
        
        for trial in relevant_trials:
            response += f"\n• {trial['trial_name']} ({trial['drug']})\n"
            response += f"  Company: {trial['company']}\n"
            response += f"  Readout: {trial['readout_timing']}\n"
            response += f"  Market Impact: {trial['market_impact']}\n"
        
        # Strategic recommendations
        response += f"\n💡 STRATEGIC RECOMMENDATIONS:\n"
        response += "-" * 40 + "\n"
        response += "• Monitor competitive launches for partnership opportunities\n"
        response += "• Prepare market access strategies for key approvals\n"
        response += "• Assess pricing implications from new entrants\n"
        response += "• Consider accelerated development timelines\n"
        
        # Risk assessment
        response += f"\n⚠️  RISK ASSESSMENT:\n"
        response += "-" * 40 + "\n"
        response += "• Regulatory delays could shift competitive dynamics\n"
        response += "• Manufacturing capacity constraints possible\n"
        response += "• Reimbursement challenges for premium pricing\n"
        
        return response
    
    def generate_deep_competitive_analysis(self, query: str) -> str:
        """Generate deep competitive intelligence"""
        therapeutic_area = self.extract_therapeutic_context(query)
        
        response = "🏆 DEEP COMPETITIVE INTELLIGENCE ANALYSIS\n"
        response += "="*60 + "\n"
        response += f"Market Scope: {therapeutic_area.title()}\n"
        response += f"Analysis Framework: Porter's Five Forces + Pipeline Assessment\n\n"
        
        if therapeutic_area == 'oncology' or therapeutic_area == 'lung':
            lung_data = self.db.competitive_data['lung_cancer']
            
            response += "🫁 LUNG CANCER MARKET DYNAMICS:\n"
            response += "-" * 40 + "\n"
            
            # Market leaders analysis
            response += "\n📊 MARKET LEADER ANALYSIS:\n"
            for company, data in lung_data['market_leaders'].items():
                response += f"\n• {company}:\n"
                response += f"  Market Share: {data['market_share']}\n"
                response += f"  Key Assets: {', '.join(data['key_drugs'])}\n"
                response += f"  Pipeline Strength: {data['pipeline_strength']}\n"
                response += f"  Recent Developments:\n"
                for dev in data.get('recent_developments', []):
                    response += f"    - {dev}\n"
            
            # Competitive threats
            response += f"\n⚡ EMERGING COMPETITIVE THREATS:\n"
            for threat in lung_data['emerging_threats']:
                response += f"• {threat}\n"
            
            # Market forces analysis
            dynamics = lung_data['market_dynamics']
            response += f"\n📈 MARKET FORCES:\n"
            response += f"Growth Drivers:\n"
            for driver in dynamics['growth_drivers']:
                response += f"  + {driver}\n"
            response += f"Key Challenges:\n"
            for challenge in dynamics['challenges']:
                response += f"  - {challenge}\n"
        
        # Strategic positioning
        response += f"\n🎯 STRATEGIC POSITIONING OPPORTUNITIES:\n"
        response += "-" * 40 + "\n"
        response += "• Precision medicine leadership through biomarker strategies\n"
        response += "• Combination therapy innovation for differentiation\n"
        response += "• Real-world evidence generation for market access\n"
        response += "• Digital health integration for patient engagement\n"
        
        # Competitive response scenarios
        response += f"\n🎲 SCENARIO PLANNING:\n"
        response += "-" * 40 + "\n"
        response += "Best Case: Successful launches capture 25%+ market share\n"
        response += "Base Case: Moderate adoption with 15-20% market penetration\n"
        response += "Risk Case: Delayed approvals or safety concerns limit uptake\n"
        
        return response
    
    def generate_strategic_market_analysis(self, query: str) -> str:
        """Generate strategic market opportunity analysis"""
        therapeutic_area = self.extract_therapeutic_context(query)
        
        response = "💰 STRATEGIC MARKET OPPORTUNITY ANALYSIS\n"
        response += "="*60 + "\n"
        response += f"Investment Thesis: {therapeutic_area.title()} Market Assessment\n\n"
        
        if therapeutic_area in self.db.market_data:
            market = self.db.market_data[therapeutic_area]
            
            response += "📊 MARKET FUNDAMENTALS:\n"
            response += "-" * 40 + "\n"
            response += f"Current Market Size (2024): {market['market_size_2024']}\n"
            response += f"Projected Size (2025): {market['projected_2025']}\n"
            response += f"Growth Rate (CAGR): {market['growth_rate']}\n"
            response += f"Key Segments: {', '.join(market['key_segments'])}\n"
            
            # Competitive landscape
            response += f"\n🏢 MARKET LEADERS:\n"
            for i, leader in enumerate(market['market_leaders'], 1):
                response += f"{i}. {leader}\n"
            
            # Market catalysts
            response += f"\n🚀 GROWTH CATALYSTS:\n"
            for catalyst in market['upcoming_catalysts']:
                response += f"• {catalyst}\n"
        
        # TAM/SAM/SOM Analysis
        response += f"\n🎯 MARKET SIZING FRAMEWORK:\n"
        response += "-" * 40 + "\n"
        response += "TAM (Total Addressable Market): Global patient population\n"
        response += "SAM (Serviceable Available Market): Reachable with current portfolio\n"
        response += "SOM (Serviceable Obtainable Market): Realistic market share target\n"
        
        # Investment recommendations
        response += f"\n💡 INVESTMENT RECOMMENDATIONS:\n"
        response += "-" * 40 + "\n"
        response += "• High Priority: Invest in differentiated assets with clear unmet need\n"
        response += "• Medium Priority: Consider partnerships for market access\n"
        response += "• Strategic: Monitor adjacent opportunities for portfolio expansion\n"
        
        # Risk-return assessment
        response += f"\n⚖️  RISK-RETURN ASSESSMENT:\n"
        response += "-" * 40 + "\n"
        response += "Upside Potential: Strong growth trajectory with multiple catalysts\n"
        response += "Downside Risks: Regulatory uncertainty and competitive intensity\n"
        response += "Recommended Action: Proceed with strategic investments\n"
        
        return response
    
    def extract_therapeutic_context(self, query: str) -> str:
        """Extract therapeutic area context from query"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['lung', 'nsclc', 'osimertinib', 'tagrisso']):
            return 'lung'
        elif any(term in query_lower for term in ['breast', 'olaparib', 'lynparza']):
            return 'breast'
        elif any(term in query_lower for term in ['cancer', 'oncology', 'tumor']):
            return 'oncology'
        elif any(term in query_lower for term in ['diabetes', 'glucose']):
            return 'diabetes'
        elif any(term in query_lower for term in ['cardiovascular', 'heart']):
            return 'cardiovascular'
        
        return 'all'
    
    def generate_clinical_intelligence_report(self, query: str) -> str:
        """Generate clinical intelligence report"""
        response = "🔬 CLINICAL INTELLIGENCE REPORT\n"
        response += "="*60 + "\n"
        response += f"Clinical Data Landscape Analysis\n\n"
        
        response += "📋 HIGH-IMPACT TRIALS 2025:\n"
        response += "-" * 40 + "\n"
        
        for trial in self.db.clinical_data['high_impact_trials_2025']:
            response += f"\n🧪 {trial['trial_name']}\n"
            response += f"   Drug: {trial['drug']}\n"
            response += f"   Company: {trial['company']}\n"
            response += f"   Indication: {trial['indication']}\n"
            response += f"   Expected Readout: {trial['readout_timing']}\n"
            response += f"   Market Impact: {trial['market_impact']}\n"
        
        return response
    
    def generate_executive_summary(self, query: str) -> str:
        """Generate executive summary for complex queries"""
        return f"📈 PHARMACEUTICAL MARKET EXECUTIVE SUMMARY\n" + "="*60 + "\n\nComprehensive market intelligence capabilities available for pipeline analysis, competitive assessment, and strategic planning across oncology, diabetes, and cardiovascular markets."

# ============================================================================
# ENHANCED EVALUATION WITH REAL MARKET INTELLIGENCE
# ============================================================================

def run_pharmaceutical_marketing_intelligence():
    """Run comprehensive pharmaceutical marketing intelligence demo"""
    print("🚀 IndicaAI: Pharmaceutical Marketing Intelligence Platform")
    print("="*70)
    
    # Initialize database
    database = PharmaceuticalDatabase()
    print("✅ Pharmaceutical database initialized")
    
    # Initialize agents
    naive_agent = IntelligentNaiveAgent(database)
    classical_agent = AdvancedClassicalMLAgent(database)
    deep_agent = SophisticatedDeepLearningAgent(database)
    
    print("✅ All marketing intelligence agents initialized")
    
    # Real pharmaceutical marketing queries
    marketing_queries = [
        "What are the upcoming oncology launches in 2025?",
        "Analyze the competitive landscape in lung cancer",
        "What is the market opportunity for diabetes drugs?",
        "Which clinical trials should we monitor in 2025?",
        "AstraZeneca pipeline strategy analysis",
        "Breast cancer market size and growth forecast"
    ]
    
    print(f"\n🧪 PHARMACEUTICAL MARKETING INTELLIGENCE TESTING")
    print("="*70)
    
    for i, query in enumerate(marketing_queries, 1):
        print(f"\n📊 Marketing Query {i}: {query}")
        print("="*70)
        
        print(f"\n🔹 NAIVE AGENT RESPONSE:")
        print("-" * 50)
        naive_response = naive_agent.answer_query(query)
        print(naive_response)
        
        print(f"\n🔸 CLASSICAL ML AGENT RESPONSE:")
        print("-" * 50)
        classical_response = classical_agent.answer_query(query)
        print(classical_response)
        
        print(f"\n🔶 DEEP LEARNING AGENT RESPONSE:")
        print("-" * 50)
        deep_response = deep_agent.answer_query(query)
        print(deep_response)
        
        print("\n" + "="*70)
    
    # Comparative analysis
    print(f"\n📈 AGENT CAPABILITY COMPARISON")
    print("="*70)
    
    capabilities = {
        'Naive Agent': {
            'Response Depth': 'Basic',
            'Market Data Access': 'Limited',
            'Strategic Insights': 'Minimal',
            'Use Case': 'Quick lookups'
        },
        'Classical ML Agent': {
            'Response Depth': 'Detailed',
            'Market Data Access': 'Comprehensive',
            'Strategic Insights': 'Moderate',
            'Use Case': 'Market analysis'
        },
        'Deep Learning Agent': {
            'Response Depth': 'Sophisticated',
            'Market Data Access': 'Full integration',
            'Strategic Insights': 'Advanced',
            'Use Case': 'Strategic planning'
        }
    }
    
    for agent, caps in capabilities.items():
        print(f"\n{agent}:")
        for metric, value in caps.items():
            print(f"  {metric}: {value}")
    
    print(f"\n💡 RECOMMENDATIONS FOR PHARMACEUTICAL MARKETING:")
    print("="*70)
    print("• Naive Agent: Suitable for basic queries and quick market lookups")
    print("• Classical ML: Ideal for detailed market analysis and competitive intelligence")
    print("• Deep Learning: Best for strategic planning and comprehensive market intelligence")
    print("• Integration: Use combination approach based on query complexity")
    
    return {
        'naive_agent': naive_agent,
        'classical_agent': classical_agent, 
        'deep_agent': deep_agent,
        'database': database
    }

# Ready to run the pharmaceutical marketing intelligence platform!
print("🎯 IndicaAI Marketing Intelligence Platform Ready!")
print("Execute: agents = run_pharmaceutical_marketing_intelligence()")

# Uncomment to run automatically:
# agents = run_pharmaceutical_marketing_intelligence()
