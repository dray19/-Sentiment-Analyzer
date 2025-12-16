#!/usr/bin/env python3
"""
Test script for sentiment analyzer application
Tests the core functionality without running Streamlit UI
"""

import sys
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def test_nltk_vader():
    """Test NLTK VADER sentiment analyzer"""
    print("=" * 60)
    print("Testing NLTK VADER Sentiment Analyzer")
    print("=" * 60)
    
    # Download NLTK data if needed
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
        print("✅ VADER lexicon found")
    except LookupError:
        print("Downloading VADER lexicon...")
        nltk.download('vader_lexicon', quiet=True)
        print("✅ VADER lexicon downloaded")
    
    # Initialize VADER
    vader = SentimentIntensityAnalyzer()
    
    # Test cases
    test_cases = [
        ("I love this amazing product! It's fantastic!", "POSITIVE"),
        ("This is the worst experience ever.", "NEGATIVE"),
        ("The item arrived on time.", "NEUTRAL"),
        ("I'm so excited about this new opportunity!", "POSITIVE"),
        ("I feel terrible about the situation.", "NEGATIVE"),
    ]
    
    passed = 0
    failed = 0
    
    for text, expected_sentiment in test_cases:
        scores = vader.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = "POSITIVE"
        elif compound <= -0.05:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        
        status = "✅" if sentiment == expected_sentiment else "❌"
        if sentiment == expected_sentiment:
            passed += 1
        else:
            failed += 1
            
        print(f"\n{status} Text: {text[:50]}...")
        print(f"   Expected: {expected_sentiment}, Got: {sentiment}")
        print(f"   Compound: {compound:.3f}, Scores: pos={scores['pos']:.3f}, neu={scores['neu']:.3f}, neg={scores['neg']:.3f}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

def test_imports():
    """Test that all required modules can be imported"""
    print("\n" + "=" * 60)
    print("Testing Module Imports")
    print("=" * 60)
    
    modules = [
        ("streamlit", "Streamlit"),
        ("transformers", "Hugging Face Transformers"),
        ("torch", "PyTorch"),
        ("nltk", "NLTK"),
    ]
    
    all_imported = True
    for module_name, display_name in modules:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
        except ImportError as e:
            print(f"❌ {display_name}: Failed to import - {e}")
            all_imported = False
    
    return all_imported

def main():
    """Run all tests"""
    print("\n🎭 Sentiment Analyzer - Test Suite\n")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test VADER
    vader_ok = test_nltk_vader()
    
    # Note about Transformer
    print("\n" + "=" * 60)
    print("Note: Hugging Face Transformer Testing")
    print("=" * 60)
    print("⚠️  Transformer model testing requires internet access")
    print("   The model will be downloaded on first use of the app")
    print("   Model: distilbert-base-uncased-finetuned-sst-2-english")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    if imports_ok and vader_ok:
        print("✅ All tests passed!")
        print("   The application is ready to run with: streamlit run app.py")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
