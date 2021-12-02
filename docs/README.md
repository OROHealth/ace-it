## Skin Lesion Classifier

### Objective:

Build a classifier model that can distinguish between 3 classes of skin disorders (Acne, Herpes simplex and Lichen planus).

### Dataset:

The initial set was composed of 112 images belonging to 3 classes: Acne (n= 40), Herpes simplex (n= 16) and Lichen planus (n= 56).  It was sourced from http://www.danderm-pdv.is.kkh.dk/atlas/index.html with permission obtained by OROHealth.

A separate test dataset was created from images downloaded from  https://www.pcds.org.uk/clinical-a-z-list, with 3 images for each of the classes in the initial dataset.

### Methodology:

Preprocessing and training were done in a Google Colab Notebook on 1 GPU (Tesla P100-PCIE-16GB), reaching a maximum GPU power use of 41W and a maximum of 9.5% GPU memory allocation.  The libraries used were PyTorch/ Torchvision/ Fastai and Wandb.

The images were verified.  One image that could not be opened was removed from the set.  The remaining images were split into a Train and a Validation set in a 80:20 ratio.  Each image was initially resized at 460.  Batch transforms included resizing to 200, rotating (max 20), zooming (max 1.2), and normalization.

A simple CNN was constructed.  The model was trained for 100 epochs, utilizing a fit-one-cycle approach to the learning rate.  The training was tracked using the Weights and Biases platform.

### Set-up for Reproduction of Results:

  1.  Clone the repository to the local machine or notebook.
     
    git clone https://github.com/yrodriguezmd/ace-it.git
    
    
  or git clonehttps://github.com/OROHealth/ace-it.git (once merged).


  2.  Using python3, create a virtual environment.

    virtualenv env
    
  3.  Install dependencies

    pip install -r requirements.txt
    
  4.  Go to ace-it directory
  
    cd ace-it
    
  5.  To train, use the folder main.py.

    python main.py
    
  6.  For inference, use the folder inference.py.
  
    python inference.py


### Results:

The highest accuracy reached was 72.7% at epoch 57.  The lowest valid_loss was 0.802 at epoch 58.  The training runtime was 00:03 minutes per epoch.


The Confusion Matrix shows the distribution of the Ground truth and Prediction labels in the Validation set, with 7 misclassifications:

![Screen Shot 2021-11-25 at 2 26 05 PM](https://user-images.githubusercontent.com/71532604/143504624-c60b7b36-990a-4644-aaaf-9230248f0903.png)


This set of images reflect the 7 misclassified labels for the Validation set at Epoch 57 (lowest validation loss):


![Screen Shot 2021-11-25 at 2 41 16 PM](https://user-images.githubusercontent.com/71532604/143505390-e3d18ac3-f885-44c3-bea0-9a1b94661c42.png)

When the model was used for inference on a held-out test set, the percentage of correct predictions was low at 33%.

### Limitations:

The dataset was small-sized and mildly unbalanced and the model architecture used was simple.

### Rooms for Improvement:

Increasing the dataset size, as well as balancing the distribution between the different classes could improve the model's accuracy, especially for identifying Herpes simplex lesions.

Increasing the batch size for the runs (once the dataset size is increased) might improve results due to a higher number of representatives seen by the model.

Utilizing more complex model architectures such as Resnet might improve the accuracy by facilitating a deeper network computation without losing the identity.

The general trend of the validation loss is still decreasing, thus a longer training run might provide better validation losses and accuracy rates.

Providing more transforms such as blur and color jitter might make the model more robust and generalizable.

Adding regularization such as drop-out to the simple CNN might improve generalizability.

### Summary:

A small, mildly unbalanced dataset composed of three skin lesion classes was used to train a simple CNN.  The resulting accuracy on the validation set was reasonable at 73%.  However, the generalizability of the model was poor.  Increasing the dataset size, improving the class balance, augmenting transformations, prolonging training duration, and utilizing a more complex network might improve the model's accuracy.


