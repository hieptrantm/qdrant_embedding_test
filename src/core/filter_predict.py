import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import os
import warnings
warnings.filterwarnings("ignore")

def predict_topic(text, model_path='filtered_model'):
    """
    Dự đoán chủ đề cho một text input
    """
    import pickle

    # Load model và tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load label encoder
    with open(os.path.join(model_path, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)

    # Tokenize input
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = predictions.argmax().item()
        confidence = predictions[0][predicted_class_id].item()

    # Decode label
    predicted_topic = label_encoder.inverse_transform([predicted_class_id])[0]

    return predicted_topic, confidence

# # Test
# test_text = "Làm thế nào để optimize hyperparameters cho neural network?"
# predicted_topic, confidence = predict_topic(test_text)
# print(f"Predicted topic: {predicted_topic}")
# print(f"Confidence: {confidence:.4f}")