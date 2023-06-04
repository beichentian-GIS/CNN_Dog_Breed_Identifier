# Use CNN to Identify Dog Breeds


## Table of Contents

1. [Installations](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#description)
4. [Results](#result)
5. [Licensing, Authors, Acknowledgements](#licensingetc)


## Installations <a name="installation"></a>
Python (version 3.8.8)

Core Python libraries needed for the analysis:

- Pandas
- Numpy
- SciKit-Learn
- Keras
- OpenCV
- TensorFlow
- Matplotlib
- Seaborn


## Project Motivations <a name="motivation"></a>
In this project, I built and trained a CNN (Convolutional Neural Networks) transfer learning model, using 8351 dog images of 133 breeds.

The trained model in this project can be further embedded into web application. When an image of a dog is provided, the model will return a predicted breed of the dog. If the input is switched to an image of a human, the model will return a dog breed that the person looks like.


## File Descriptions <a name="description"></a>
	- README.md: project brief displayed on Github
	- dog_app.ipynb: scripts to construct and train CNN model to identify dog breeds
	- extract_bottleneck_features.py: functions to compute bottleneck features given a tensor converted from an image
	- \bottleneck_features
		DogResnet50Data.npz: a pre-computed the bottleneck features for ResNet50 using dog image data
	- \haarcascades
		haarcascade_frontalface_alt.xml:  a pre-trained face detector provided by OpenCV
	- \images
		image files used as input to test the model performance
	- \saved_models
		weights.best.Resnet50.hdf5: saved model weights with best validation loss
		Resnet50.h5: saved model architecture


## Results <a name="result"></a>
The insights of the analyses can be found at [this blog post](https://medium.com/@btiangis91/use-cnn-to-identify-dog-breeds-2ff542e589a4) on Medium.


## Licensing, Authors, Acknowledgements <a name="licensingetc"></a>
Credits are given to Udacity for the data and education.