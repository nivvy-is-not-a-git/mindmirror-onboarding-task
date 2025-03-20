## MindMirror Onboarding Assignment: AI Prototype Development

### Introduction

Welcome to the MindMirror team! In this onboarding assignment, you will build a practical prototype focusing on two core AI-powered features that are fundamental to our journaling app:

1. **Emotion Detection** – automatically identifying emotions from user journal entries.  
2. **Text Summarization** – generating concise summaries to help users reflect quickly and effectively.

This assignment aims to familiarize you with core Natural Language Processing (NLP) techniques through practical experience, preparing you for more complex tasks within our project.

---

## Setup and Preprocessing (Before You Begin)

Before implementing your models, take time to set up your environment and prepare your input data. This step ensures that the models you choose will work smoothly with the text inputs.

### Purpose
Preprocessing is an essential step in any ML workflow. Even when using pre-trained models or ready-to-use pipelines, input data may need cleaning, normalization, or segmentation. Journal entries are often informal, unstructured, and vary in length—making light preprocessing especially important in our context.

### Instructions
1. **Install Required Libraries**  
   Create a virtual environment (optional but recommended), and install necessary Python libraries. You may need some of the libaries below, depending on how you plan on creating the two features:
   ```bash
   pip install transformers torch pandas scikit-learn
   ```

2. **Collect or Create Sample Journal Entries**  
   Gather or write at least 10 realistic journal entries. Vary them in tone, length, and content. These entries will be used throughout this assignment for testing both emotion detection and summarization.  
   > This exercise simulates a real-world task we will face during development: sourcing and curating appropriate data for model training and evaluation.

3. **Preprocess the Text**  
   - Remove unnecessary whitespace, emojis, or formatting artifacts.
   - Ensure the text is in sentence form (not fragments or bullet points).
   - If a journal entry is very long (e.g., over 1024 tokens), split it into smaller segments that can be processed by the model.
   - (Optional) Lowercase the text if your selected model benefits from case normalization.

4. **Prepare for Batch Input**  
   - Store your entries in a list of strings or a CSV/JSON format for easy iteration.
   - Keep them organized for reusability across both tasks.

---

## Task 1: Emotion Detection

In this task, you will design a pipeline that can identify emotions expressed in user-written journal entries.

### Purpose
Emotion detection is crucial for MindMirror as it provides users with valuable insights into their emotional states, assisting them in identifying patterns and trends. NLP models pre-trained on emotion-labeled datasets can be used to recognize these emotional signals from natural language input.

Instead of providing you with a model of our choosing, we ask you to research available pre-trained emotion classification models using Hugging Face’s model hub or similar repositories. Consider models trained on relevant datasets (e.g., GoEmotions, EmotionLines) and evaluate them based on performance, speed, and compatibility with our journaling use case.

### Instructions

1. **Research and select a suitable pre-trained emotion classification model.**  
   Document your reasoning: why did you choose this model? What dataset was it trained on? How many emotion classes does it support?

2. **Implement a pipeline** that processes multiple journal entries, applies your selected model, and returns structured emotion predictions for each entry.

3. **Document and reflect.**  
   For each prediction, assess whether the output emotions make sense. If not, speculate on possible reasons for misclassification (e.g., ambiguous language, overlapping emotion categories).

### Example Output Format
```json
[
  {
    "journal_entry": "Today was stressful, but completing my assignments felt rewarding.",
    "emotions": ["stress", "joy"]
  },
  {
    "journal_entry": "Feeling down today, missing home.",
    "emotions": ["sadness"]
  }
]
```
### Tips & Common Pitfalls

Emotion labels from models can be overly general or highly specific—standardize if needed.

- Watch out for low-confidence predictions—filter or threshold scores may help.

- Multi-label classification is common for emotions; avoid forcing single-label output unless the model is designed for it.

- Ambiguous or mixed-emotion entries may yield unpredictable results—annotate them carefully.

---

## Task 2: Text Summarization

In this task, you will implement automatic text summarization for journal entries. Summarization helps users quickly reflect on key experiences without needing to reread lengthy journal entries, thereby enhancing the user experience and engagement.

### Purpose
Summarization models condense large amounts of text into concise, meaningful narratives. These models are typically trained on large sum of documents and their corresponding summaries. For our use case, the summarization feature provides users with reflective insights, condensing their daily entries into a sentence or two that captures emotional and experiential highlights.

As with Task 1, we encourage you to explore and select a suitable pre-trained summarization model. Consider model size, performance, and input/output capabilities.

### Instructions

1. **Research and select a summarization model.**  
   Compare available models and select one that you believe best suits the journaling context. Document why you chose this model.

2. **Implement a summarization function.**  
   The function should take a journal entry as input and output a summary that reflects the core idea and emotional highlights.

3. **Evaluate the summaries.**  
   For each output, manually assess how well it represents the original entry. Are key points retained? Is the emotional tone preserved? Provide at least 3 examples of both strong and weak summaries, along with your analysis.

### Example
**Journal Entry:**
```
I had a busy day with numerous meetings. Initially felt overwhelmed, but finishing my important tasks provided relief. Later, dinner with friends was very enjoyable and relaxing.
```

**Summary:**
```
Your day started stressful due to meetings but ended positively after task completion and quality time with friends.
```
### Tips & Common Pitfalls

- Summarization models may miss emotional nuance—evaluate tone, not just content.

- Entries with little narrative structure may produce vague or unhelpful summaries.

- Extremely short or long entries can break model expectations—use length control parameters or segmentation.

- Repetitive phrasing in input text may confuse the model; lightly clean the input if needed.
---

## Starter Code Template (Python)

We have provided a structured template to guide your implementation. You do not have to use this template if you do not want to, as it is a simple boilerplate to get you started.

---

## Submission Guidelines

To submit your completed assignment, please follow this GitHub-based workflow. We assume you have basic knowledge of Git & GitHub:

1. **Fork the main onboarding repository**: You will receive a link to the MindMirror onboarding repo. Fork it to your own GitHub account.

2. **Work in your forked repository**:
   - Create a new folder under `submissions/your-name/` (use your first name or GitHub username).
   - Place all your code files, sample data, and write-up (README or notebook) inside this folder.

3. **Document your work** in a clear and organized manner. Your submission folder should include:
   - Completed emotion detection & summarization implementations
   - Any data you generated or sourced
   - A short explanation of model selection and reasoning
   - Reflections, observations, and any challenges encountered

4. **Submit your work via Pull Request**:
   - Open a pull request from your fork to the `main` branch of the onboarding repo.
   - Please ensure you have a comment block at the top of your submitted code with your name and your Waterloo username.

This assignment is your opportunity to get hands-on with the tools and techniques we’ll be using throughout the MindMirror project. Treat it as both a learning exercise and a demonstration of your approach to solving applied AI problems.

Good luck, and happy building!

