# Handwritten Character Recognition

This project implements a Handwritten Character Recognition system using TensorFlow and Keras. The model is trained to recognize handwritten names from images, leveraging Convolutional Neural Networks (CNNs) and Bidirectional LSTM layers. It uses a Kaggle dataset for training and evaluation.

## Key Features

- **Data Loading & Visualization**: Load and visualize handwritten images from the Kaggle dataset.
- **Data Preprocessing**: Resize, normalize images, and handle labels for training.
- **Model Architecture**: Utilizes CNN for feature extraction and Bidirectional LSTM for sequence prediction with Connectionist Temporal Classification (CTC) loss.
- **Model Training**: Trained on preprocessed data with validation checks.
- **Evaluation**: Assesses model performance on the validation set and provides accuracy metrics.
- **Real-time Predictions**: Interface for predicting handwritten names using Gradio.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/handwritten-character-recognition.git
    cd handwritten-character-recognition
    ```

2. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset from Kaggle:
    ```bash
    pip install opendatasets
    import opendatasets as od
    od.download("https://www.kaggle.com/datasets/landlord/handwriting-recognition/code")
    ```

## Usage

1. Run the training script:
    ```bash
    python train.py
    ```

2. Use the Gradio interface to make predictions:
    ```bash
    python app.py
    ```

## Files

- `train.py`: Script for training the handwritten character recognition model.
- `app.py`: Gradio application for real-time predictions.
- `model/`: Directory containing the saved model.
- `requirements.txt`: List of required Python packages.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Kaggle Dataset](https://www.kaggle.com/datasets/landlord/handwriting-recognition)
- [TensorFlow & Keras Documentation](https://www.tensorflow.org/)

