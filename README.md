# Gender-Prediction-Deep-Learning
The projects works on Gender Prediction as a classification problem. The output layer in the gender prediction network is of type softmax with 2 nodes indicating the two classes “Male” and “Female”.

The network uses 3 convolutional layers, 2 fully connected layers and a final output node. The details of the layers are given below.
Conv1 : The first convolutional layer has 32 nodes of kernel with relu activation function.
Conv2 : The second conv layer has 32 nodes with kernel with relu activation function
and one output layer with 1 node and activation function sigmoid.


Libraries Required:
1. keras
2. numpy
3. h5py(To save the model)
4. Tenser flow at the backend of keras.
5. Theno


The dataset is available at: https://www.kaggle.com/playlist/men-women-classification/version/3#

