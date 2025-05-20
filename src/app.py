import gradio as gr
from fastai.vision.all import *
import torch

# Load the fastai model
try:
    learn = load_learner('invasive_weed_multiclass_classifier.pkl')
except Exception as e:
    raise Exception(f"Failed to load model: {e}")

# Define prediction function
def classify_weed(image):
    try:
        # Ensure model is on CPU (Hugging Face free tier uses CPU)
        learn.model = learn.model.cpu()
        # Predict
        pred, pred_idx, probs = learn.predict(image)
        # Create dictionary of class probabilities
        prob_dict = {learn.dls.vocab[i]: float(probs[i]) for i in range(len(learn.dls.vocab))}
        # Format output
        result = f"Predicted Class: {pred}\n\nProbabilities:\n"
        for cls, prob in prob_dict.items():
            result += f"{cls}: {prob:.4f}\n"
        return result
    except Exception as e:
        return f"Error during prediction: {e}"

# Create Gradio interface
demo = gr.Interface(
    fn=classify_weed,
    inputs=gr.Image(type="pil", label="Upload a weed image"),
    outputs=gr.Textbox(label="Prediction Result"),
    title="DeepWeeds Classifier",
    description="Upload an image to classify it as one of 8 weed species or 'Negative'. Trained on the DeepWeeds dataset using ResNet18.",
    examples=[
        "https://storage.googleapis.com/kaggle-datasets-images/149999/267423/5b0f7c8e9f7a7c9d9b9c9f9b9c9f9c9f/dataset-cover.jpg"  # Example image URL
    ],
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    demo.launch()