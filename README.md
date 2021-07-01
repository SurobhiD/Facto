# # Gesture Recognition Using CNN and Transfer Learning

## **Problem Statement :** 
Imagine you are working as a data scientist at a home electronics company which manufactures state of the art smart televisions. You want to develop a cool feature in the smart-TV that can recognise five different gestures performed by the user which will help users control the TV without using a remote.

The gestures are continuously monitored by the webcam mounted on the TV. Each gesture corresponds to a specific command:
 
| Gesture | Corresponding Action |
| --- | --- | 
| Thumbs Up | Increase the volume. |
| Thumbs Down | Decrease the volume. |
| Left Swipe | 'Jump' backwards 10 seconds. |
| Right Swipe | 'Jump' forward 10 seconds. |
| Stop | Pause the movie. |

Each video is a sequence of 30 frames (or images).

 ![alt text](https://i.pinimg.com/originals/dc/c8/b2/dcc8b2876e12282bddcabcca1e091bd0.jpg)

### **Python Libraries Used** 

Here is a list of Python Libraries used along with their version numbers :

 |Library                         |   Version                         |
|-------------------------------|-----------------------------|
|   `imageio`           |    2.6.1          |
|`Numpy`            |1.18.1           |
|`pandas`|1.0.1|
|`tensorflow`|2.3.2|
|`Keras`|2.4.3|

### **Input Datasets** 
The training data consists of a few hundred videos categorized into one of the five classes. Each video (typically 2-3 seconds long) is divided into a sequence of 30 frames(images). These videos have been recorded by various people performing one of the five gestures in front of a webcam - similar to what the smart TV will use. 

The data  file contains a 'train' and a 'val' folder with two CSV files for the two folders. These folders are in turn divided into sub-folders where each sub-folder represents a video of a particular gesture. Each sub-folder, i.e. a video, contains 30 frames (or images). Note that all images in a particular video sub-folder have the same dimensions but different videos may have different dimensions. Specifically, videos have two types of dimensions - either 360x360 or 120x160 (depending on the webcam used to record the videos). Hence, we will need to do some pre-processing to standardise the videos. 

Each row of the CSV file represents one video and contains three main pieces of information - the name of the sub-folder containing the 30 images of the video, the name of the gesture and the numeric label (between 0-4) of the video.

We need to train a model on the 'train' folder which performs well on the 'val' folder as well (as usually done in ML projects). 

### **Implementation in Python Notebook** 
#### *Using the Generator Function*

In the generator, we pre-process the images as we have images of 2 different dimensions and also create a batch of video frames. We will fix the image size to be 120X120 pixels and consider only 6 frames for every video. We will also normalize the images.
Note here that a video is represented above in the generator as (number of images, height, width, number of channels). Take this into consideration while creating the model architecture.

In all we have 663 training sequences and 100 validation sequences.

#### *Model Building & training :*
Here we will make the model using different functionalities that Keras provides. The network should be designed  in such a way that the model is able to give good accuracy on the least number of parameters so that it can fit in the memory of the webcam.

We will go ahead and create 17 different models , each using different deep learning techniques and model parameters and evaluate the results. Here is a look-up of the final results :

![table1](https://user-images.githubusercontent.com/10894854/124152179-ee2da480-dab0-11eb-98f9-18f2cb53d632.JPG)
![table2](https://user-images.githubusercontent.com/10894854/124152187-eff76800-dab0-11eb-9212-bb4609bb2a76.JPG)
![table3](https://user-images.githubusercontent.com/10894854/124152195-f1289500-dab0-11eb-9f44-0a1744bbfbae.JPG)
![table4](https://user-images.githubusercontent.com/10894854/124152199-f259c200-dab0-11eb-9164-a18c5b0497f9.JPG)
![table5](https://user-images.githubusercontent.com/10894854/124152206-f38aef00-dab0-11eb-809e-e70b0fd2c770.JPG)
We reached a training accuracy of 86.9% and a validation accuracy of 84%. The Validation Accuracy has improved considerably and the gap between the training and validation accuracy has got bridged, and it stands at just 2-3% now.


