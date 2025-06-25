#!/usr/bin/env python3
"""
Complete pipeline runner for IndicaAI
Creates sample data, processes it, and tests all models
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the complete pipeline"""
    print("🚀 IndicaAI Complete Pipeline")
    print("=" * 50)

    # Step 1: Create sample data
    print("📊 Step 1: Creating sample data...")
    try:
        from scripts.make_dataset import create_all_sample_data
        sample_data = create_all_sample_data()
        total_records = sum(len(df) for df in sample_data.values())
        print(f"✅ Created {total_records} records across {len(sample_data)} datasets")
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return

    # Step 2: Process the data
    print("\n🔄 Step 2: Processing data...")
    try:
        from scripts.data_processing import DataProcessor
        processor = DataProcessor()
        processed_data = processor.process_all_data()

        if processed_data:
            print("✅ Data processing completed!")
            print("📈 Processed datasets:")
            for data_type, df in processed_data.items():
                print(f"  {data_type}: {len(df)} records")
        else:
            print("⚠️  No data was processed")
            return
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return

    # Step 3: Test naive model
    print("\n🧪 Step 3: Testing Naive Model...")
    try:
        from models.naive_model import NaivePharmaceuticalAgent

        agent = NaivePharmaceuticalAgent()
        agent.load_data()

        test_queries = [
            "What are the side effects of aspirin?",
            "Tell me about metformin for diabetes",
            "Clinical trials for ibuprofen pain relief"
        ]

        for query in test_queries:
            print(f"\n❓ Query: {query}")
            response = agent.answer_query(query)
            print(f"🤖 Response: {response}")

        print("✅ Naive model testing completed!")

    except Exception as e:
        print(f"❌ Error testing naive model: {e}")

    # Step 4: Test classical ML
    print("\n🔬 Step 4: Testing Classical ML...")
    try:
        from models.classical_ml import ClassicalMLAgent

        agent = ClassicalMLAgent()
        results = agent.train_models()

        if results:
            print("✅ Classical ML training completed!")
            print("📊 Training results:")
            for model_type, metrics in results.items():
                print(f"  {model_type}: {len(metrics)} classifiers trained")
        else:
            print("⚠️  Classical ML training had issues")

    except Exception as e:
        print(f"❌ Error with classical ML: {e}")

    # Step 5: Launch Streamlit app
    print("\n🖥️  Step 5: Ready to launch Streamlit app!")
    print("Run: streamlit run main.py")
    print("\n🎯 Pipeline completed successfully!")
    print("All three approaches are ready for evaluation.")

if __name__ == "__main__":
    main()
