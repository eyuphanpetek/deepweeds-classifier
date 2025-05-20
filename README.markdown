# DeepWeeds Classifier

A deep learning project to classify invasive weeds into 9 classes (8 weed species + "Negative") using the [DeepWeeds dataset](https://www.kaggle.com/datasets/imsparsh/deepweeds). The model is trained with **fastai** and **ResNet18** in **Google Colab**, achieving ~87–89% accuracy, and deployed as an interactive web app on **Hugging Face Spaces** using **Gradio**.

## Project Overview
- **Objective**: Build a multi-class classifier for weed identification to support sustainable agriculture.
- **Dataset**: DeepWeeds (17,509 images, 9 classes).
- **Training**: Optimized ResNet18 model, trained in ~1.5–2.5 hours on Colab’s T4 GPU.
- **Deployment**: Gradio app on Hugging Face Spaces for real-time predictions.
- **Outputs**: Visualizations (batch preview, confusion matrix, top losses).

## Repository Structure
```
deepweeds-classifier/
├── models/
│   └── invasive_weed_multiclass_classifier.pkl  # Trained model
├── outputs/
│   ├── deepweeds_multiclass_batch_preview.png    # Batch visualization
│   ├── deepweeds_multiclass_top_losses.png       # Top losses
│   └── deepweeds_multiclass_confusion_matrix.png # Confusion matrix
├── src/
│   ├── train_deepweeds.ipynb          # Training script
│   └── app.py                         # Gradio app
├── requirements.txt                   # Dependencies
├── README.md                          # This file
└── LICENSE                            # MIT License
```

## Setup and Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/eyuphanpetek/deepweeds-classifier.git
   cd deepweeds-classifier
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Install Git LFS** (for the model file):
   ```bash
   git lfs install
   git lfs pull
   ```

## Usage
### Running the Gradio App Locally
1. Navigate to the `src/` folder:
   ```bash
   cd src
   ```
2. Run the app:
   ```bash
   python app.py
   ```
3. Open the local URL (e.g., `http://127.0.0.1:7860`) in a browser.
4. Upload an image to classify it as one of the 9 weed classes.

### Try the Live App
Visit the deployed app on Hugging Face Spaces: [https://huggingface.co/spaces/eyuphanpetek/deepweeds_classifier_epetek_01](https://huggingface.co/spaces/eyuphanpetek/deepweeds_classifier_epetek_01).

### Training the Model
1. Open `src/train_deepweeds.ipynb` in Jupyter Notebook or Google Colab.
2. Follow the instructions to download the DeepWeeds dataset and train the model.
3. Requirements: Colab with T4 GPU, Kaggle API key.

## Results
- **Accuracy**: ~87–89% on the validation set.
- **Runtime**: ~1.5–2.5 hours on Colab T4 GPU.
- **Visualizations**:
  - Batch preview: `outputs/deepweeds_multiclass_batch_preview.png`
  - Confusion matrix: `outputs/deepweeds_multiclass_confusion_matrix.png`
  - Top losses: `outputs/deepweeds_multiclass_top_losses.png`

## Blog Post
Read the full project journey on Medium: [[Here](https://medium.com/@eyuphanpetek/classifying-invasive-weeds-with-deep-learning-yet-another-classification-problem-c14ed1d9c08b)].

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments
- DeepWeeds dataset: [https://www.kaggle.com/datasets/imsparsh/deepweeds](https://www.kaggle.com/datasets/imsparsh/deepweeds)
- fastai: [https://docs.fast.ai/](https://docs.fast.ai/)
- Gradio: [https://www.gradio.app/](https://www.gradio.app/)
- Hugging Face Spaces: [https://huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)

## Contact
Eyüphan Petek  
TED University Computer Engineering    
Email: [eyuphan.petek@gmail.com]
