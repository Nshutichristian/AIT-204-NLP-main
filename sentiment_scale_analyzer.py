"""
Multi-Scale Sentiment Analyzer: -3 to +3
Students: Complete the TODOs to implement 7-point sentiment scale
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')


def class_to_sentiment_score(class_id):
    """
    TODO 3: Convert class ID (0-6) to sentiment score (-3 to +3)

    Hint: If class_id ranges from 0 to 6, and we want -3 to +3,
    what mathematical operation would you use?

    Args:
        class_id: Integer from 0 to 6
    Returns:
        sentiment_score: Integer from -3 to +3
    """
    # TODO: Write the conversion formula here
    sentiment_score = class_id - 3  # SOLUTION: Remove this line and let students figure it out

    return sentiment_score


def get_sentiment_label(score):
    """Convert numeric score to descriptive label"""
    labels = {
        -3: "Very Negative 😢",
        -2: "Negative 😞",
        -1: "Slightly Negative 😐",
        0: "Neutral 😶",
        1: "Slightly Positive 🙂",
        2: "Positive 😊",
        3: "Very Positive 🤩"
    }
    return labels.get(score, "Unknown")


def create_training_data():
    """
    TODO 2: Create labeled training data for 7-point scale

    Add more examples for each sentiment level.
    Format: (review_text, sentiment_class)
    where sentiment_class: 0=Very Negative, 3=Neutral, 6=Very Positive
    """

    training_data = [
        # Very Negative (-3) → Class 0
        ("This is the worst movie I have ever seen in my entire life!", 0),
        ("Absolutely terrible! Complete waste of time and money.", 0),
        ("Horrible film. I want my money back. Awful in every way.", 0),
        # TODO: Add 2 more very negative examples here

        # Negative (-2) → Class 1
        ("This movie was quite disappointing and boring.", 1),
        ("Not good. Poor acting and weak plot.", 1),
        # TODO: Add 2 more negative examples here

        # Slightly Negative (-1) → Class 2
        ("The movie had potential but didn't deliver.", 2),
        ("It was below average, not terrible but not good either.", 2),
        # TODO: Add 2 more slightly negative examples here

        # Neutral (0) → Class 3
        ("It was okay, nothing special. Just average.", 3),
        ("The film was fine. Neither good nor bad.", 3),
        # TODO: Add 2 more neutral examples here

        # Slightly Positive (+1) → Class 4
        ("Pretty decent movie. I enjoyed it overall.", 4),
        ("Good film with some nice moments.", 4),
        # TODO: Add 2 more slightly positive examples here

        # Positive (+2) → Class 5
        ("Really great movie! I thoroughly enjoyed it.", 5),
        ("Excellent film with strong performances.", 5),
        # TODO: Add 2 more positive examples here

        # Very Positive (+3) → Class 6
        ("Absolutely amazing! Best movie I've seen this year!", 6),
        ("Masterpiece! Incredible storytelling and perfect execution.", 6),
        ("This movie was phenomenal! A true work of art!", 6),
        # TODO: Add 2 more very positive examples here
    ]

    return training_data


def analyze_sentiment(text, model, tokenizer, device):
    """
    Analyze sentiment of a single text using the trained model

    Args:
        text: Review text to analyze
        model: Trained model
        tokenizer: Tokenizer
        device: CPU or GPU
    Returns:
        sentiment_score: -3 to +3
        confidence: 0 to 1
        sentiment_label: Descriptive label
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = probabilities.argmax(dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

    # Convert to sentiment score
    sentiment_score = class_to_sentiment_score(predicted_class)
    sentiment_label = get_sentiment_label(sentiment_score)

    return sentiment_score, confidence, sentiment_label


def main():
    print("=" * 80)
    print("Multi-Scale Sentiment Analyzer: -3 to +3")
    print("=" * 80)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Using device: {device}")

    # TODO 1: Load model with correct number of labels
    print("\n🤖 Loading model...")
    model_name = "distilbert-base-uncased"

    # TODO: Modify num_labels for 7-point scale (-3 to +3)
    # How many classes do we need? Fill in the blank below:
    num_labels = 7  # SOLUTION: Students should figure out this is 7

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True  # Allow different output size
        )
        model.to(device)
        print("✓ Model loaded successfully!")
        print(f"  - Model: {model_name}")
        print(f"  - Number of sentiment classes: {num_labels}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nNote: If you see a network error, the model needs to be downloaded first.")
        print("Make sure you have internet access or use a pre-downloaded model.")
        return

    # Show training data structure
    print("\n📊 Training Data Structure:")
    training_data = create_training_data()
    print(f"  - Total examples: {len(training_data)}")

    # Count examples per class
    from collections import Counter
    class_counts = Counter([label for _, label in training_data])
    print("\n  Examples per sentiment level:")
    for class_id in sorted(class_counts.keys()):
        score = class_to_sentiment_score(class_id)
        label = get_sentiment_label(score)
        count = class_counts[class_id]
        print(f"    Class {class_id} (Score {score:+d}): {label} - {count} examples")

    # Note: In a real implementation, you would train the model here
    # For this activity, we'll use the pre-trained model for demonstration
    print("\n⚠️  Note: For the 1-hour activity, we're using the pre-trained model")
    print("    In production, you would fine-tune on the labeled data above.")

    # Test with example reviews
    print("\n🧪 Testing Multi-Scale Sentiment Analysis")
    print("=" * 80)

    test_reviews = [
        "This movie was absolutely phenomenal! Best film of the decade!",
        "Really enjoyed this film. Great performances all around.",
        "Pretty good movie, had some nice moments.",
        "It was okay, nothing particularly memorable.",
        "The film had potential but fell short.",
        "Quite disappointing. Expected much better.",
        "Absolutely terrible! Worst movie I've ever seen!",
    ]

    for i, review in enumerate(test_reviews, 1):
        print(f"\n{i}. Review: \"{review}\"")

        # Analyze sentiment
        score, confidence, label = analyze_sentiment(review, model, tokenizer, device)

        # Display results
        print(f"   Sentiment Score: {score:+d}/3")
        print(f"   Label: {label}")
        print(f"   Confidence: {confidence:.1%}")

        # Visual representation
        visual_bar = "█" * (score + 3) + "░" * (3 - score)
        print(f"   Scale: [{visual_bar}] ({score:+d})")

    # Interactive testing
    print("\n" + "=" * 80)
    print("🎮 Interactive Mode: Test Your Own Reviews")
    print("=" * 80)
    print("Enter movie reviews to analyze (or 'quit' to exit)\n")

    while True:
        user_input = input("Enter review: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thanks for using the Multi-Scale Sentiment Analyzer!")
            break

        if not user_input:
            continue

        # Analyze
        score, confidence, label = analyze_sentiment(user_input, model, tokenizer, device)

        # Display
        print(f"\n  → Sentiment Score: {score:+d}/3")
        print(f"  → Label: {label}")
        print(f"  → Confidence: {confidence:.1%}\n")


if __name__ == "__main__":
    main()
