"""
STUDENT NAME: Avin Shetty
STUDENT ID: a26shett
"""

from transformers import pipeline

# Load your chosen models here
def load_emotion_model():
    # Research and return a pre-trained emotion detection pipeline
    classifier= pipeline (task="sentiment-analysis",model="SamLowe/roberta-base-go_emotions", top_k=5)
    return classifier

def load_summarization_model():
    # Research and return a pre-trained summarization pipeline
    summarizer= pipeline ("summarization", model="facebook/bart-large-cnn")
    return summarizer

# Process journal entries and return emotion predictions
def detect_emotions(sentences, classifier):
    # Implement logic to process entries and extract emotions
    all_predictions=[]
    for sentence in sentences:
        all_predictions.append (classifier(sentence))
    return all_predictions

# Generate summaries for journal entries
def summarize_entries(entry, summarizer):
    # Implement logic to summarize each entry
    summarized_entry=summarizer(entry,max_length=50, min_length=30, do_sample=False)
    return summarized_entry

if __name__ == '__main__':
    # Load models
    emotion_model = load_emotion_model()
    summarizer = load_summarization_model()

    # Example input
    emotional_entries = [
        "I love you",
        "I really love going to physics class! I also really love when the physics professor makes us do a lot of boring work and then yells at us!",
        "I saw my grandmother today. She’s getting older, and I can see it more each time I visit. She held my hand and told me stories from her childhood, laughing like she was still that little girl. I don’t think she realizes how much I cherish these moments. One day, I’ll tell my kids about her, but for now, I just want to sit with her a little longer.",
        "I was at a café today, just watching people come and go. It’s funny how much you can learn from body language. There was a woman staring at her phone, biting her lip—bad news, maybe?",
        "Why do people make things so much harder than they need to be? I spent all week preparing for this meeting, only for my manager to completely ignore half of what I said. I know I’m not imagining it—I work just as hard, but my ideas always get dismissed. Maybe it’s time I stop playing nice and start making them listen.",
        "I drove past my old high school today, and it hit me—how much time has passed, how much I’ve changed. I used to think those years were everything, like every decision was life or death. Now, I barely recognize the person I was back then. I wish I could go back and tell my younger self: It gets better. The things you’re worried about now won’t even matter in a few years.",
        "You know what? I think I’m finally getting my life together."
]
    long_entries=[
        "Today, I took a moment to really appreciate everything I have. Life isn’t perfect, and there are always challenges, but when I step back, I realize how lucky I am. I have a roof over my head, food on my table, and people who care about me. It’s so easy to focus on what’s missing or what’s going wrong, but today, I chose to focus on what’s right. I went for a walk and felt grateful for the fresh air, the warmth of the sun, and even the little things—like my favorite song playing at just the right moment. I had a good conversation with a friend, which reminded me how important it is to have people who listen and support you. Even the small moments, like enjoying a quiet cup of coffee, felt more meaningful. Life moves so fast, and I don’t always take the time to appreciate what I already have, but today, I did—and it felt good."
    ]

    # Apply pipelines
    emotions = detect_emotions(emotional_entries, emotion_model)
    summaries = summarize_entries(long_entries, summarizer)

    # Output results

    for i, sentence_output in enumerate(emotions):
    # sentence_output[0] is now the actual list of dicts: [{'label': 'love', 'score': 0.94}, ...]
        predictions = sentence_output[0]

        print(f"\nSentence {i+1} results:")
        for item in predictions:
            print(f"  {item['label']}: {item['score']:.4f}")

    print("Summaries:", summaries)