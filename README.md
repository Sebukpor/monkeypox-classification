
##MonkeyPox Classification Model

This project involves training a convolutional neural network (CNN) to classify images as either "MonkeyPox" or "Other" skin conditions. The model is built using TensorFlow and Keras, leveraging a pre-trained ResNet50 model to achieve high accuracy with a small dataset.

Table of Contents
Installation
Dataset
Model Architecture
Training
Evaluation
Usage
Contributing
License
Installation
Clone the repository:


Copy code
git clone https://github.com/sebukpor/monkeypox-classification.git
cd monkeypox-classification
Install required dependencies:




Copy code
python monkey_pox.py
Dataset
The dataset used for this project contains images of skin conditions classified into two categories: "MonkeyPox" and "Other". The images are stored in separate folders for each class.

Dataset Structure:

markdown
Copy code
Original Images/
    ├── MonkeyPox/
    └── Other/
The dataset metadata is stored in Monkeypox_Dataset_metadata.csv.

Image Preprocessing: The images are resized to 300x300 pixels, and data augmentation techniques such as rescaling, shearing, zooming, and horizontal flipping are applied.

Model Architecture
Pre-trained Model: ResNet50 is used as the base model with pre-trained weights from ImageNet.
Custom Layers:
A flattening layer to convert the 2D matrix to a 1D vector.
A dense layer with 512 units and ReLU activation.
A final dense layer with 2 units (for "MonkeyPox" and "Other") using softmax activation.
Training
Optimizer: SGD with a learning rate of 0.0009.
Loss Function: Sparse Categorical Crossentropy.
Epochs: 35
Training Data: 80% of the dataset is used for training.
Validation Data: 20% of the dataset is used for validation.
The model is trained using the training set, and its performance is validated on a separate validation set. Accuracy and loss metrics are plotted to monitor the training process.

Evaluation
After training, the model's accuracy is evaluated on the validation set. The script also includes functionality to test the model on a single image and predict its class.

Usage
To use the trained model for prediction:

Load an image:

python
Copy code
img = load_img('path_to_image.jpg', target_size=(300, 300))
Preprocess the image:

python
Copy code
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
Predict the class:

python
Copy code
y_pred = resnet_model.predict(x)
class_idx = np.argmax(y_pred, axis=1)[0]
class_name = ['MonkeyPox', 'Other'][class_idx]
print('Predicted Class name:', class_name)
Contributing
Contributions are welcome! Please fork this repository, make your changes, and submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

