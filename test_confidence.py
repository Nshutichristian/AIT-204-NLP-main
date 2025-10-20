"""
Quick test script to verify confidence scores are working correctly
"""

# Test reviews with expected confidence ranges
test_cases = [
    {
        "review": "This movie is absolutely phenomenal! A masterpiece of cinema. The acting is superb, the story is captivating, and the cinematography is breathtaking. Every scene is perfectly crafted. This is the best film I've seen in years! Highly recommended to everyone!",
        "expected_sentiment": "positive",
        "expected_confidence_min": 0.85,
        "description": "Very strong positive"
    },
    {
        "review": "Absolutely horrible! One of the worst movies ever made. Terrible acting, nonsensical plot, and complete waste of time and money. Save yourself and don't watch this garbage!",
        "expected_sentiment": "negative",
        "expected_confidence_min": 0.85,
        "description": "Very strong negative"
    },
    {
        "review": "Really enjoyed this movie. Great performances and an interesting plot. Would definitely watch it again.",
        "expected_sentiment": "positive",
        "expected_confidence_min": 0.70,
        "description": "Moderate positive"
    },
    {
        "review": "Very disappointed with this film. The story was poorly written and the pacing was terrible.",
        "expected_sentiment": "negative",
        "expected_confidence_min": 0.70,
        "description": "Moderate negative"
    },
    {
        "review": "The movie had some good moments but also some weak parts. Not bad, not great.",
        "expected_sentiment": "positive",  # May vary
        "expected_confidence_min": 0.50,
        "description": "Mixed review (low confidence expected)"
    }
]

print("=" * 80)
print("CONFIDENCE SCORE TEST RESULTS")
print("=" * 80)
print("\nTest cases designed to verify confidence scores range properly")
print("Expected ranges:")
print("  - Very strong sentiment: 85-99%")
print("  - Moderate sentiment: 70-85%")
print("  - Weak/Mixed sentiment: 50-70%")
print("\n" + "=" * 80)

for i, test in enumerate(test_cases, 1):
    print(f"\nTest {i}: {test['description']}")
    print(f"Review: \"{test['review'][:80]}...\"")
    print(f"Expected: {test['expected_sentiment']} with confidence >= {test['expected_confidence_min']*100:.0f}%")
    print("Status: ✓ Ready to test (run Streamlit app to test)")

print("\n" + "=" * 80)
print("To test: Run the Streamlit app and paste these reviews into the analyzer")
print("=" * 80)
