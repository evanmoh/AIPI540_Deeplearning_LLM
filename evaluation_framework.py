#!/usr/bin/env python3
"""
Pharmaceutical Marketing Intelligence Model Evaluation Framework
NO HARDCODED RESPONSES - Uses actual model outputs only
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass
from enum import Enum
import time

# ============================================================================
# EVALUATION FRAMEWORK CLASSES
# ============================================================================

class EvaluationMetricType(Enum):
    TECHNICAL = "technical"
    BUSINESS = "business" 
    CLINICAL = "clinical"
    STRATEGIC = "strategic"

@dataclass
class EvaluationResult:
    metric_name: str
    score: float
    max_score: float
    category: EvaluationMetricType
    details: Dict
    recommendations: List[str]

class PharmaceuticalModelEvaluator:
    """Evaluation framework for pharmaceutical marketing intelligence models"""
    
    def __init__(self):
        self.setup_evaluation_criteria()
        self.setup_benchmark_data()
        
    def setup_evaluation_criteria(self):
        """Define evaluation criteria"""
        
        self.technical_metrics = {
            'accuracy': {'weight': 0.25, 'description': 'Classification accuracy'},
            'response_relevance': {'weight': 0.20, 'description': 'Response relevance'},
            'entity_extraction': {'weight': 0.15, 'description': 'Entity extraction accuracy'},
            'response_time': {'weight': 0.10, 'description': 'Response speed'},
            'consistency': {'weight': 0.30, 'description': 'Response consistency'}
        }
        
        self.business_metrics = {
            'market_data_accuracy': {'weight': 0.30, 'description': 'Market data accuracy'},
            'competitive_intelligence_quality': {'weight': 0.25, 'description': 'Competitive insights'},
            'pipeline_intelligence': {'weight': 0.20, 'description': 'Pipeline accuracy'},
            'strategic_insight_depth': {'weight': 0.25, 'description': 'Strategic insights'}
        }
        
        self.clinical_metrics = {
            'clinical_terminology_usage': {'weight': 0.50, 'description': 'Clinical terminology'},
            'safety_information_inclusion': {'weight': 0.50, 'description': 'Safety information'}
        }
        
        self.strategic_metrics = {
            'strategic_insight_quality': {'weight': 0.40, 'description': 'Strategic insights'},
            'decision_support_quality': {'weight': 0.35, 'description': 'Decision support'},
            'risk_assessment_capability': {'weight': 0.25, 'description': 'Risk assessment'}
        }
    
    def setup_benchmark_data(self):
        """Setup benchmark queries and expected results"""
        
        self.benchmark_queries = [
            {
                'query': 'What are the upcoming oncology launches in 2025?',
                'query_type': 'pipeline_analysis',
                'therapeutic_area': 'oncology',
                'expected_entities': {
                    'drugs': ['Dato-DXd', 'Capivasertib'],
                    'companies': ['AstraZeneca', 'Daiichi Sankyo'],
                    'timeframe': '2025'
                },
                'evaluation_criteria': {
                    'must_include': ['2025', 'launch', 'oncology'],
                    'should_include': ['AstraZeneca', 'pipeline', 'drug'],
                    'accuracy_benchmarks': {
                        'timeframe': '2025',
                        'therapeutic_area': 'oncology'
                    }
                }
            },
            {
                'query': 'Analyze the competitive landscape in lung cancer',
                'query_type': 'competitive_analysis',
                'therapeutic_area': 'oncology',
                'expected_entities': {
                    'companies': ['AstraZeneca', 'Merck'],
                    'drugs': ['Tagrisso', 'Keytruda'],
                    'therapeutic_area': ['lung cancer']
                },
                'evaluation_criteria': {
                    'must_include': ['competitive', 'lung cancer'],
                    'should_include': ['AstraZeneca', 'Merck', 'market share'],
                    'accuracy_benchmarks': {
                        'therapeutic_area': 'lung cancer'
                    }
                }
            },
            {
                'query': 'What is the market opportunity for diabetes drugs?',
                'query_type': 'market_analysis',
                'therapeutic_area': 'diabetes',
                'expected_entities': {
                    'therapeutic_area': ['diabetes'],
                    'market_concepts': ['opportunity', 'market', 'size']
                },
                'evaluation_criteria': {
                    'must_include': ['market', 'diabetes'],
                    'should_include': ['opportunity', 'growth', 'size'],
                    'accuracy_benchmarks': {
                        'therapeutic_area': 'diabetes'
                    }
                }
            }
        ]
    
    def evaluate_model_comprehensive(self, model, model_name: str) -> Dict:
        """Comprehensive evaluation of a pharmaceutical model"""
        
        print(f"\n🔍 COMPREHENSIVE EVALUATION: {model_name}")
        print("="*60)
        
        results = {
            'model_name': model_name,
            'overall_score': 0,
            'category_scores': {},
            'detailed_results': {},
            'recommendations': []
        }
        
        # Evaluate all categories
        technical_results = self.evaluate_technical_performance(model)
        business_results = self.evaluate_business_value(model)
        clinical_results = self.evaluate_clinical_intelligence(model)
        strategic_results = self.evaluate_strategic_support(model)
        
        results['category_scores']['technical'] = technical_results
        results['category_scores']['business'] = business_results
        results['category_scores']['clinical'] = clinical_results
        results['category_scores']['strategic'] = strategic_results
        
        # Calculate overall score
        category_weights = {'technical': 0.25, 'business': 0.35, 'clinical': 0.20, 'strategic': 0.20}
        overall_score = sum(results['category_scores'][cat]['overall_score'] * weight 
                           for cat, weight in category_weights.items())
        results['overall_score'] = overall_score
        
        # Generate recommendations
        results['recommendations'] = self.generate_improvement_recommendations(results)
        
        return results
    
    def evaluate_technical_performance(self, model) -> Dict:
        """Evaluate technical performance metrics"""
        
        print("\n📊 Technical Performance Evaluation")
        print("-" * 40)
        
        technical_scores = {}
        
        correct_classifications = 0
        total_queries = len(self.benchmark_queries)
        response_times = []
        entity_extraction_scores = []
        
        for query_data in self.benchmark_queries:
            query = query_data['query']
            
            # Measure response time
            start_time = time.time()
            
            try:
                response = model.answer_query(query)
                response_time = (time.time() - start_time) * 1000
                response_times.append(response_time)
                
                # Check entity extraction
                entity_score = self.evaluate_entity_extraction(response, query_data['expected_entities'])
                entity_extraction_scores.append(entity_score)
                
                # Check query classification
                if hasattr(model, 'classify_query_intent'):
                    predicted_intent = model.classify_query_intent(query)
                    if predicted_intent == query_data['query_type']:
                        correct_classifications += 1
                else:
                    # Check if response is relevant to query type
                    if self.check_response_relevance(response, query_data['query_type']):
                        correct_classifications += 1
                
            except Exception as e:
                print(f"Error evaluating query: {e}")
                response_times.append(5000)
                entity_extraction_scores.append(0)
        
        # Calculate scores
        technical_scores['accuracy'] = {
            'score': correct_classifications / total_queries,
            'details': f"{correct_classifications}/{total_queries} correct classifications"
        }
        
        technical_scores['response_time'] = {
            'score': max(0, 1 - (np.mean(response_times) / 1000)),
            'details': f"Average: {np.mean(response_times):.1f}ms"
        }
        
        technical_scores['entity_extraction'] = {
            'score': np.mean(entity_extraction_scores) if entity_extraction_scores else 0,
            'details': f"Average entity extraction: {np.mean(entity_extraction_scores):.2f}"
        }
        
        # Calculate weighted technical score
        total_weight = sum(self.technical_metrics[metric]['weight'] for metric in technical_scores.keys())
        overall_technical_score = sum(
            technical_scores[metric]['score'] * self.technical_metrics[metric]['weight']
            for metric in technical_scores.keys()
        ) / total_weight if total_weight > 0 else 0
        
        return {
            'overall_score': overall_technical_score,
            'detailed_scores': technical_scores,
            'category': 'technical'
        }
    
    def evaluate_business_value(self, model) -> Dict:
        """Evaluate business value metrics"""
        
        print("\n💼 Business Value Evaluation")
        print("-" * 40)
        
        business_scores = {}
        
        market_accuracy_scores = []
        competitive_quality_scores = []
        pipeline_accuracy_scores = []
        
        for query_data in self.benchmark_queries:
            response = model.answer_query(query_data['query'])
            
            if query_data['query_type'] == 'market_analysis':
                accuracy_score = self.validate_market_data_accuracy(response, query_data)
                market_accuracy_scores.append(accuracy_score)
                
            elif query_data['query_type'] == 'competitive_analysis':
                quality_score = self.evaluate_competitive_intelligence_quality(response, query_data)
                competitive_quality_scores.append(quality_score)
                
            elif query_data['query_type'] == 'pipeline_analysis':
                pipeline_score = self.validate_pipeline_intelligence(response, query_data)
                pipeline_accuracy_scores.append(pipeline_score)
        
        business_scores['market_data_accuracy'] = {
            'score': np.mean(market_accuracy_scores) if market_accuracy_scores else 0,
            'details': f"Market data validation across {len(market_accuracy_scores)} queries"
        }
        
        business_scores['competitive_intelligence_quality'] = {
            'score': np.mean(competitive_quality_scores) if competitive_quality_scores else 0,
            'details': f"Competitive analysis quality across {len(competitive_quality_scores)} queries"
        }
        
        business_scores['pipeline_intelligence'] = {
            'score': np.mean(pipeline_accuracy_scores) if pipeline_accuracy_scores else 0,
            'details': f"Pipeline data accuracy across {len(pipeline_accuracy_scores)} queries"
        }
        
        # Calculate overall business score
        total_weight = sum(self.business_metrics[metric]['weight'] for metric in business_scores.keys())
        overall_business_score = sum(
            business_scores[metric]['score'] * self.business_metrics[metric]['weight']
            for metric in business_scores.keys()
        ) / total_weight if total_weight > 0 else 0
        
        return {
            'overall_score': overall_business_score,
            'detailed_scores': business_scores,
            'category': 'business'
        }
    
    def evaluate_clinical_intelligence(self, model) -> Dict:
        """Evaluate clinical intelligence based on real model responses"""
        
        print("\n🔬 Clinical Intelligence Evaluation")
        print("-" * 40)
        
        clinical_scores = {}
        
        clinical_queries = [
            "What are the side effects of osimertinib?",
            "Clinical trial data for breast cancer treatments",
            "Safety profile of diabetes medications"
        ]
        
        clinical_term_scores = []
        safety_mention_scores = []
        
        clinical_terms = ['clinical', 'trial', 'efficacy', 'safety', 'adverse', 'patient', 'study', 'data']
        safety_terms = ['side effect', 'adverse event', 'safety', 'toxicity', 'contraindication', 'warning']
        
        for query in clinical_queries:
            try:
                response = model.answer_query(query)
                response_lower = response.lower()
                
                # Score clinical terminology usage
                clinical_term_count = sum(1 for term in clinical_terms if term in response_lower)
                clinical_term_score = min(clinical_term_count / 4, 1.0)
                clinical_term_scores.append(clinical_term_score)
                
                # Score safety information inclusion
                safety_term_count = sum(1 for term in safety_terms if term in response_lower)
                safety_score = min(safety_term_count / 2, 1.0)
                safety_mention_scores.append(safety_score)
                
            except Exception as e:
                clinical_term_scores.append(0)
                safety_mention_scores.append(0)
        
        clinical_scores['clinical_terminology_usage'] = {
            'score': np.mean(clinical_term_scores) if clinical_term_scores else 0,
            'details': f'Clinical terminology usage across {len(clinical_queries)} queries'
        }
        
        clinical_scores['safety_information_inclusion'] = {
            'score': np.mean(safety_mention_scores) if safety_mention_scores else 0,
            'details': f'Safety information inclusion across {len(clinical_queries)} queries'
        }
        
        overall_clinical_score = np.mean([score['score'] for score in clinical_scores.values()])
        
        return {
            'overall_score': overall_clinical_score,
            'detailed_scores': clinical_scores,
            'category': 'clinical'
        }
    
    def evaluate_strategic_support(self, model) -> Dict:
        """Evaluate strategic decision support based on real response analysis"""
        
        print("\n🎯 Strategic Decision Support Evaluation")
        print("-" * 40)
        
        strategic_scores = {}
        
        strategic_queries = [
            "What strategic recommendations do you have for oncology investment?",
            "How should we position against competitors in diabetes market?",
            "What are the key risks in our pipeline strategy?"
        ]
        
        decision_support_scores = []
        strategic_insight_scores = []
        risk_assessment_scores = []
        
        strategic_indicators = ['strategy', 'recommendation', 'competitive', 'risk', 'opportunity', 'investment']
        decision_indicators = ['should', 'recommend', 'suggest', 'consider', 'evaluate', 'prioritize']
        risk_indicators = ['risk', 'challenge', 'threat', 'concern', 'mitigation', 'contingency']
        
        for i, query in enumerate(strategic_queries):
            try:
                response = model.answer_query(query)
                response_lower = response.lower()
                
                # Score strategic content
                strategic_count = sum(1 for indicator in strategic_indicators if indicator in response_lower)
                strategic_score = min(strategic_count / 3, 1.0)
                strategic_insight_scores.append(strategic_score)
                
                # Score decision support quality
                decision_count = sum(1 for indicator in decision_indicators if indicator in response_lower)
                decision_score = min(decision_count / 2, 1.0)
                decision_support_scores.append(decision_score)
                
                # Score risk assessment (only for risk-related query)
                if i == 2:
                    risk_count = sum(1 for indicator in risk_indicators if indicator in response_lower)
                    risk_score = min(risk_count / 2, 1.0)
                    risk_assessment_scores.append(risk_score)
                
            except Exception as e:
                strategic_insight_scores.append(0)
                decision_support_scores.append(0)
                if i == 2:
                    risk_assessment_scores.append(0)
        
        strategic_scores['strategic_insight_quality'] = {
            'score': np.mean(strategic_insight_scores) if strategic_insight_scores else 0,
            'details': f'Strategic insight analysis across {len(strategic_queries)} queries'
        }
        
        strategic_scores['decision_support_quality'] = {
            'score': np.mean(decision_support_scores) if decision_support_scores else 0,
            'details': f'Decision support evaluation across {len(strategic_queries)} queries'
        }
        
        if risk_assessment_scores:
            strategic_scores['risk_assessment_capability'] = {
                'score': np.mean(risk_assessment_scores),
                'details': f'Risk assessment quality in strategic responses'
            }
        
        overall_strategic_score = np.mean([score['score'] for score in strategic_scores.values()])
        
        return {
            'overall_score': overall_strategic_score,
            'detailed_scores': strategic_scores,
            'category': 'strategic'
        }
    
    def evaluate_entity_extraction(self, response: str, expected_entities: Dict) -> float:
        """Evaluate entity extraction accuracy"""
        
        response_lower = response.lower()
        total_score = 0
        total_categories = 0
        
        for category, entities in expected_entities.items():
            if entities:
                total_categories += 1
                found_entities = sum(1 for entity in entities if str(entity).lower() in response_lower)
                category_score = found_entities / len(entities)
                total_score += category_score
        
        return total_score / total_categories if total_categories > 0 else 0
    
    def check_response_relevance(self, response: str, query_type: str) -> bool:
        """Check if response is relevant to query type"""
        
        response_lower = response.lower()
        
        if query_type == 'pipeline_analysis':
            return any(term in response_lower for term in ['launch', 'pipeline', 'drug', 'approval'])
        elif query_type == 'competitive_analysis':
            return any(term in response_lower for term in ['competitive', 'competitor', 'market share', 'landscape'])
        elif query_type == 'market_analysis':
            return any(term in response_lower for term in ['market', 'opportunity', 'size', 'growth'])
        
        return True
    
    def validate_market_data_accuracy(self, response: str, query_data: Dict) -> float:
        """Validate market data accuracy against benchmarks"""
        
        must_include = query_data.get('evaluation_criteria', {}).get('must_include', [])
        response_lower = response.lower()
        
        score = sum(1 for term in must_include if term.lower() in response_lower)
        return score / len(must_include) if must_include else 0
    
    def evaluate_competitive_intelligence_quality(self, response: str, query_data: Dict) -> float:
        """Evaluate competitive intelligence quality"""
        
        must_include = query_data.get('evaluation_criteria', {}).get('must_include', [])
        should_include = query_data.get('evaluation_criteria', {}).get('should_include', [])
        
        response_lower = response.lower()
        
        must_score = sum(1 for term in must_include if term.lower() in response_lower) / len(must_include) if must_include else 0
        should_score = sum(1 for term in should_include if term.lower() in response_lower) / len(should_include) if should_include else 0
        
        overall_score = (must_score * 0.7) + (should_score * 0.3)
        return overall_score
    
    def validate_pipeline_intelligence(self, response: str, query_data: Dict) -> float:
        """Validate pipeline intelligence accuracy"""
        return self.validate_market_data_accuracy(response, query_data)
    
    def generate_improvement_recommendations(self, results: Dict) -> List[str]:
        """Generate improvement recommendations based on real evaluation results"""
        
        recommendations = []
        
        # Get actual scores
        technical_score = results['category_scores']['technical']['overall_score']
        business_score = results['category_scores']['business']['overall_score']
        clinical_score = results['category_scores']['clinical']['overall_score']
        strategic_score = results['category_scores']['strategic']['overall_score']
        overall_score = results['overall_score']
        
        # Generate specific recommendations
        if technical_score < 0.7:
            recommendations.append(f"Technical performance ({technical_score:.1%}) needs improvement - focus on response accuracy")
        
        if business_score < 0.7:
            recommendations.append(f"Business intelligence ({business_score:.1%}) needs enhancement - improve market data accuracy")
        
        if clinical_score < 0.7:
            recommendations.append(f"Clinical intelligence ({clinical_score:.1%}) requires improvement - increase clinical terminology usage")
        
        if strategic_score < 0.7:
            recommendations.append(f"Strategic capabilities ({strategic_score:.1%}) need development - enhance decision support quality")
        
        # Overall recommendation
        if overall_score >= 0.85:
            recommendations.append(f"Excellent performance ({overall_score:.1%}) - ready for production deployment")
        elif overall_score >= 0.75:
            recommendations.append(f"Strong performance ({overall_score:.1%}) - suitable for pilot deployment")
        elif overall_score >= 0.65:
            recommendations.append(f"Moderate performance ({overall_score:.1%}) - requires targeted improvements")
        else:
            recommendations.append(f"Performance ({overall_score:.1%}) below production threshold - significant improvements needed")
        
        return recommendations
    
    def print_evaluation_report(self, results: Dict):
        """Print comprehensive evaluation report"""
        
        print(f"\n📋 EVALUATION REPORT: {results['model_name']}")
        print("="*60)
        
        overall_score = results['overall_score']
        print(f"\n🎯 OVERALL SCORE: {overall_score:.1%}")
        
        if overall_score >= 0.8:
            grade = "🟢 EXCELLENT"
        elif overall_score >= 0.7:
            grade = "🟡 GOOD"
        elif overall_score >= 0.6:
            grade = "🟠 FAIR"
        else:
            grade = "🔴 POOR"
            
        print(f"Performance Grade: {grade}")
        
        # Category scores
        print(f"\n📊 CATEGORY BREAKDOWN:")
        print("-" * 40)
        
        for category, category_results in results['category_scores'].items():
            score = category_results['overall_score']
            print(f"{category.title():20} {score:.1%}")
            
            for metric, details in category_results['detailed_scores'].items():
                print(f"  {metric:18} {details['score']:.1%} - {details['details']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*60)

# ============================================================================
# EVALUATION EXECUTION FUNCTIONS
# ============================================================================

def run_comprehensive_model_evaluation(models_dict):
    """Run comprehensive evaluation on all models"""
    
    print("🔍 COMPREHENSIVE PHARMACEUTICAL MODEL EVALUATION")
    print("="*70)
    
    evaluator = PharmaceuticalModelEvaluator()
    all_results = {}
    
    for model_name, model in models_dict.items():
        print(f"\n🧪 Evaluating {model_name}...")
        
        try:
            results = evaluator.evaluate_model_comprehensive(model, model_name)
            all_results[model_name] = results
            evaluator.print_evaluation_report(results)
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            continue
    
    # Comparative analysis
    print(f"\n🏆 COMPARATIVE ANALYSIS")
    print("="*70)
    
    if len(all_results) > 1:
        print(f"\n📊 OVERALL PERFORMANCE RANKING:")
        print("-" * 40)
        
        sorted_models = sorted(all_results.items(), 
                             key=lambda x: x[1]['overall_score'], 
                             reverse=True)
        
        for i, (model_name, results) in enumerate(sorted_models, 1):
            score = results['overall_score']
            print(f"{i}. {model_name:20} {score:.1%}")
        
        # Best in category
        print(f"\n🥇 BEST IN CATEGORY:")
        print("-" * 40)
        
        categories = ['technical', 'business', 'clinical', 'strategic']
        for category in categories:
            best_model = max(all_results.items(), 
                           key=lambda x: x[1]['category_scores'][category]['overall_score'])
            best_score = best_model[1]['category_scores'][category]['overall_score']
            print(f"{category.title():15} {best_model[0]:20} {best_score:.1%}")
    
    # Summary recommendations
    print(f"\n📈 DEPLOYMENT RECOMMENDATIONS:")
    print("-" * 40)
    
    for model_name, results in all_results.items():
        overall_score = results['overall_score']
        
        if overall_score >= 0.8:
            status = "✅ PRODUCTION READY"
        elif overall_score >= 0.7:
            status = "🔶 PILOT DEPLOYMENT"
        elif overall_score >= 0.6:
            status = "🔄 IMPROVEMENT NEEDED"
        else:
            status = "❌ NOT RECOMMENDED"
            
        print(f"{model_name:20} {status}")
    
    return all_results

def evaluate_real_pharmaceutical_models_with_agents(agents_dict):
    """Evaluate using the actual pharmaceutical agents you created"""
    
    print("🔍 EVALUATING YOUR ACTUAL PHARMACEUTICAL MODELS")
    print("="*60)
    
    # Extract the real agents
    naive_agent = agents_dict.get('naive_agent')
    classical_agent = agents_dict.get('classical_agent') 
    deep_agent = agents_dict.get('deep_agent')
    
    if not all([naive_agent, classical_agent, deep_agent]):
        print("❌ Error: Missing agents in the provided dictionary")
        print("Expected keys: 'naive_agent', 'classical_agent', 'deep_agent'")
        return None
    
    # Test the actual models with real queries
    test_queries = [
        "What are the upcoming oncology launches in 2025?",
        "Analyze the competitive landscape in lung cancer",
        "What is the market opportunity for diabetes drugs?",
        "What are the side effects of osimertinib?",
        "Strategic recommendations for pipeline investment"
    ]
    
    print(f"\n📊 Testing actual model responses:")
    print("-" * 60)
    
    # Show actual responses from your models
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔹 Query {i}: {query}")
        print("=" * 50)
        
        try:
            print(f"\n🤖 NAIVE AGENT (Your Actual Model):")
            naive_response = naive_agent.answer_query(query)
            print(f"Response: {naive_response}")
            
            print(f"\n🤖 CLASSICAL ML AGENT (Your Actual Model):")
            classical_response = classical_agent.answer_query(query)
            print(f"Response: {classical_response}")
            
            print(f"\n🤖 DEEP LEARNING AGENT (Your Actual Model):")
            deep_response = deep_agent.answer_query(query)
            print(f"Response: {deep_response}")
            
        except Exception as e:
            print(f"❌ Error getting response: {e}")
        
        print("\n" + "-"*60)
    
    # Now evaluate the actual models
    evaluator = PharmaceuticalModelEvaluator()
    
    models_to_evaluate = {
        'Your Naive Agent': naive_agent,
        'Your Classical ML Agent': classical_agent,
        'Your Deep Learning Agent': deep_agent
    }
    
    print(f"\n🔍 Running Comprehensive Evaluation on Your Actual Models...")
    results = run_comprehensive_model_evaluation(models_to_evaluate)
    
    return results

# ============================================================================
# READY TO USE
# ============================================================================

print("🎯 Pharmaceutical Model Evaluation Framework Ready!")
print("\nUsage:")
print("1. First create your agents: agents = run_pharmaceutical_marketing_intelligence()")
print("2. Then evaluate: results = evaluate_real_pharmaceutical_models_with_agents(agents)")
print("3. This will use your ACTUAL models with NO hardcoded responses!")
