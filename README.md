# Infant Cry Detection using Causal Temporal Representation

This project focuses on detecting infant cries using a novel **causal temporal representation** framework. Our approach incorporates causal reasoning into the data-generating process (DGP) to improve the interpretability and reliability of cry detection systems. This repository provides the necessary resources to explore, train, and evaluate supervised models for this task, along with mathematical assumptions and metrics tailored for event-based evaluation.

## Features
- **Data Generating Process**: Based on mathematical causal assumptions, our DGP defines how audio features and annotations are causally connected.
- **Supervised Models**: State-of-the-art supervised learning methods, including Bidirectional LSTM, Transformer, and MobileNet V2.
- **Event-Based Metrics**: Evaluation metrics tailored for time-sensitive detection tasks, including event-based F1-score and IOU.
- **Interactive Example**: A Jupyter Notebook with step-by-step demonstrations.

![Data Generating Process](main.img)

## Repository Structure

```plaintext
.
├── data/               # Audio data in .wav format
├── labels/             # Annotation files corresponding to audio data (.TextGrid)
├── metrics/            # Event-based evaluation metrics
├── models/             # Pre-trained supervised models
├── src/                # Core codebase
├── experiment.ipynb    # Usage demonstration
└── README.md           # Project description
```

### Directory Details

- **data/**: Contains raw audio files in `.wav` format.
  - Each audio file represents an infant cry recording.

- **labels/**: Stores annotation files in `.TextGrid` format.
  - Each `.TextGrid` file corresponds to an audio file and provides ground truth segmentations for cry events.

- **metrics/**: Houses the implementation of event-based metrics for evaluating the performance of models.
  - Metrics include event-based F1-score and IOU, designed to measure temporal accuracy effectively.

- **models/**: Contains pre-trained supervised models for infant cry detection.
  - Models include:
    - Bidirectional LSTM
    - Transformer
    - MobileNet V2

- **src/**: Core implementation of the infant cry detection framework.
  - Includes modules for data preprocessing, feature extraction, model training, and evaluation.

- **experiment.ipynb**: A Jupyter Notebook with a simple use case example.
  - Demonstrates how to load data, preprocess it, train a model, and evaluate its performance.

For more details, refer to our accompanying research paper.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
