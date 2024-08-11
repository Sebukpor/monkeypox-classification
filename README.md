
# MonkeyPox Classification Using CNN with ResNet50

This project involves building a deep learning model to classify images of skin conditions as either "MonkeyPox" or "Other." The model is based on a convolutional neural network (CNN) architecture using the ResNet50 model pre-trained on ImageNet. The goal is to leverage transfer learning to achieve high accuracy even with a limited dataset.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/monkeypox-classification.git
    cd monkeypox-classification
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the script:

    ```bash
    python monkey_pox.py
    ```

## Dataset

The dataset used in this project contains images of skin conditions categorized into two classes: "MonkeyPox" and "Other." The images are stored in separate folders according to their class labels.

### Dataset Structure:

```plaintext
Original Images/
    ├── MonkeyPox/
    └── Other/
```

The dataset metadata is stored in `Monkeypox_Dataset_metadata.csv`.

### Image Preprocessing

- The images are resized to 300x300 pixels.
- Data augmentation techniques such as rescaling, shearing, zooming, and horizontal flipping are applied to enhance model generalization.

## Model Architecture

- **Pre-trained Model**: The ResNet50 model is used as the base model with pre-trained weights from ImageNet.
- **Custom Layers**:
  - A flattening layer to convert the 2D matrix to a 1D vector.
  - A dense layer with 512 units and ReLU activation.
  - A final dense layer with 2 units (for "MonkeyPox" and "Other") using softmax activation.

## Training

- **Optimizer**: Stochastic Gradient Descent (SGD) with a learning rate of 0.0009.
- **Loss Function**: Sparse Categorical Crossentropy.
- **Epochs**: 35
- **Training Data**: 80% of the dataset is used for training.
- **Validation Data**: 20% of the dataset is used for validation.

The model is trained on the training set and validated on a separate validation set. Accuracy and loss metrics are plotted to monitor the training process.

## Evaluation

After training, the model’s accuracy is evaluated on the validation set. The script also includes functionality to test the model on a single image and predict its class.

## Usage

To use the trained model for prediction:

1. Load an image:

    ```python
    from tensorflow.keras.utils import load_img, img_to_array

    img = load_img('path_to_image.jpg', target_size=(300, 300))
    ```

2. Preprocess the image:

    ```python
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    ```

3. Predict the class:

    ```python
    y_pred = resnet_model.predict(x)
    class_idx = np.argmax(y_pred, axis=1)[0]
    class_name = ['MonkeyPox', 'Other'][class_idx]
    print('Predicted Class name:', class_name)
    ```

## Contributing

Contributions are welcome! Please fork this repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to customize this README further according to your specific needs.
