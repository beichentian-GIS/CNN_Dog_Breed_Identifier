# Use CNN to Identify Dog Breeds


## Table of Contents

1. [Installations](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#description)
4. [Conclusion](#conclusion)
5. [Future Improvements](#improvement)
6. [Licensing, Authors, Acknowledgements](#licensingetc)


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
In this project, a CNN (Convolutional Neural Networks) model was built for identifying dog breed for a given image of dog. 
In order to train this model, 8,351 images of dogs of 133 breeds were used. 
In addition, pre-computed bottleneck features based on ResNet50 network was embedded into the model to achieve better prediction accuracy as well as higher training efficiency. 
There are three goals/ questions that motivate the development of such model:
- Given an image of dog, can the model be able to accurately determine its breed?
- Given an image of human (face), can the model be able to provide a dog breed that resembles the facial characteristics of the human?
- Given an image of neither a dog nor a human, can the model be able to tell the state and provide a warning message?


## File Descriptions <a name="description"></a>
	- README.md: project brief displayed on Github
	- dog_app.ipynb: scripts to construct and train CNN model to identify dog breeds
	- extract_bottleneck_features.py: functions to compute bottleneck features given a tensor converted from an image
	- \bottleneck_features
		DogResnet50Data.npz: pre-computed bottleneck features for ResNet50 using dog image data
	- \haarcascades
		haarcascade_frontalface_alt.xml:  a pre-trained face detector provided by OpenCV
	- \images
		image files used as input to test the model performance
	- \saved_models
		weights.best.Resnet50.hdf5: saved model weights with best validation loss
		Resnet50.h5: saved model architecture


## Conclusion <a name="conclusion"></a>
The model (embedded with pre-trained ResNet50 bottleneck features) reached an overall test accuracy of 83.85%.
The model's predicted dog breeds sometimes do not exactly match the actual dog breeds yet are under the same breed categories. 
The occurrence of such issue may be introduced by variations in amounts of images across dog breeds. For two breeds under the same category, it is more likely for the model to provide an answer of the breed with more training images. 
Other factors that may cause inaccuracy prediction may include but are not limited to the model’s potential defects, the quality of the given images such as resolution, brightness, etc.
More details can be found at [this blog post](https://medium.com/@btiangis91/use-cnn-to-identify-dog-breeds-v2-a11acfd79038) on Medium.


## Future Improvements <a name="improvement"></a>
- Increase the amount of training images to no less than 50 or 60 for each dog breed whose original training images are less than 40.
- The model parameters (number of layers, number of nodes, dropout threshold, etc.) may be adjusted to achieve a better result.
- Segment human face or dog out of the image background using [SAM](https://segment-anything.com/) (Segment Anything Model) to have the model only focused on what truly needs to be predicted.
- Apply data augmentation techniques consisted of image translations, horizontal reflections, and mean subtraction to artificially provide more features to train.


## Licensing, Authors, Acknowledgements <a name="licensingetc"></a>
Credits are given to Udacity for the data and education.