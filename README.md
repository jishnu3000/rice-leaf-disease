# Rice Leaf Disease Detection

This project focuses on classifying rice leaf diseases using computer vision and deep learning. Leveraging convolutional neural networks (CNNs) and transfer learning, this model aims to accurately distinguish between three common rice leaf diseases: Bacterial Leaf Blight, Brown Spot, and Leaf Smut. 

## Project Code: PRCP-1001-RiceLeaf

## Techniques Demonstrated:

- **Image Processing and Data Augmentation**
  - Loading and pre-processing images for analysis
  - Data augmentation techniques to increase dataset size and diversity

- **Convolutional Neural Networks (CNN)**
  - Building a custom CNN model from scratch to classify images
  - Using layers such as Conv2D, MaxPooling, Batch Normalization, and Dropout for feature extraction and generalization
  
- **Transfer Learning with Pre-trained Models**
  - Experimenting with three popular pre-trained models (e.g., VGG16, ResNet, MobileNet) to improve accuracy and reduce training time

- **Evaluation and Metrics**
  - Evaluating model performance using metrics like accuracy, precision, recall, and F1-score
  - Visualizing training and validation accuracy/loss over epochs

- **Visualization and Results**
  - Displaying sample predictions and comparing model performance
  - Analyzing confusion matrix to evaluate classification success per class

## Dependencies

This project uses Python and several libraries, including:
- `numpy`, `pandas` for data handling
- `matplotlib`, `seaborn` for data visualization
- `keras` and `tensorflow` for building and training neural networks
- `PIL` and `splitfolders` for image handling and dataset splitting

## Results

| Model Name             |   Train Accuracy Percentage |   Train Loss |   Validation Accuracy Percentage |   Validation Loss |   Test Accuracy Percentage |   Test Loss |
|:-----------------------|----------------------------:|-------------:|---------------------------------:|------------------:|---------------------------:|------------:|
| CNN Model from Scratch |                     94.7598 |       0.2431 |                          96.0995 |            0.2003 |                    95.5063 |      0.2323 |
| EfficientNetB0         |                     99.3533 |       0.0302 |                          99.8655 |            0.0035 |                    99.9329 |      0.0007 |
| InceptionV3            |                     94.1299 |       0.2058 |                          97.7135 |            0.0798 |                    97.7135 |      0.0798 |
| Xception               |                     98.8915 |       0.0331 |                          99.8655 |            0.0046 |                    99.8658 |      0.0058 |
