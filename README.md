
# English to Amharic Translation Model

This project aims to build a neural machine translation model to translate text from English to Amharic using Recurrent Neural Networks (RNNs).

## Language Selection

The target language for translation is Amharic.

## Dataset Preparation

### Dataset Quality

The dataset contains English sentences and their corresponding translations in Amharic. The sentences are diverse and represent common usage patterns.

### Data Files

- `data/english_sentences.txt`: Contains English sentences.
- `data/amharic_sentences.txt`: Contains Amharic translations of the English sentences.

## Model Building

### Model Architecture

The model utilizes a Recurrent Neural Network (RNN) with Bidirectional Long Short-Term Memory (LSTM) layers.

### Preprocessing Steps

1. **Tokenization**: Convert text into sequences of integers.
2. **Padding**: Pad sequences to ensure uniform input lengths.

## Training

The model is trained using the prepared dataset. Key hyperparameters include:
- Embedding dimension: 100
- LSTM units: 128
- Batch size: 32
- Epochs: 10

## Evaluation

The model's performance is evaluated using the BLEU score, which measures the accuracy of the translation compared to the reference translations.

## Requirements

The project requires the following packages:
- TensorFlow
- NLTK
- NumPy

Install the required packages using:
```bash
pip install -r requirements.txt
