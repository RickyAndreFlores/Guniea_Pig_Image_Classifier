# Guniea_Pig_Image_Classifier

**Peanut Classifier:**

The Peanut Classifier is an image classifier trained to classify Guinea Pigs that I made. It was built using machine learning, more specifically it was built using a neural network. This neural network is loosely based off AlexNet, once one of the fastest algorithms for computer vision, back when it came out in 2012. It has then been surpassed by other architectures such as Residual Networks, which I plan on implementing soon. 


**Structure:** 

conv1          convolution and rectified linear activation.    

pool1          max pooling.   
norm1          local response normalization.  

conv2          convolution and rectified linear activation.  
norm2          local response normalization.  
pool2          max pooling.  

local3         fully connected layer with rectified linear activation.  
local4         fully connected layer with no activation.  
softmax        linear transformation to produce logits.  
  

**Goals:**

1. To be able to input my own collected data in the form of images and preprocess them to be useful in training a CNN. In this 		simplified program this means, being  able to read the image files and convert them to usable matrices of the same size (batch size, X,  Y, depth/channels)

2. To be able to both implement the entire neural network from start to finish using the Tensorflow framework and visualize the results using Tensorboard. This means programming all the relevant code into computation graphs, their respective tensors, and executing them with Tensorflow.

3. To make a functional algorithm that can classify Guinea Pigs given a certain image. To define a metric, it would be for the algorithm to classify >50% of the training data correct. Additionally it would be good to have the cross validation data to be classified correctly over 50% of the time as well. 


**Background on the Project Name:**

This model was trained to classify pictures of Guinea Pigs. It was inspired by my fiancés love and care for our pet guinea pigs. Out of the bunch was one guinea pig named was Peanut.  This little guinea pig was there for my fiancé since she was young and was a very nice companion. Sadly, Peanut passed away recently, and so this Neural Network was named after him as a respectful nod to his memory. In a way, this program will theoretically be able to remember Peanut way past our own lifetimes, a happy thought that hopefully brings some joy to other pet owners.

Below is an image of peanut   
<img src="https://raw.githubusercontent.com/RickyAndreFlores/Guniea_Pig_Image_Classifier/master/Peanut%20-%20Copy.JPG" height="60">


**Results:**    

**83% accuracy** on training data after 100 iterations, with a **53% accuracy** on the cross validation data. 


**Conclusion and Potential Improvements:**

In conclusion, the Neural Network works, but it is clear that it is over fit onto the training data. A quick solution for is to provide the Neural Network with much more, and more diverse data. 

The source of these downsides come down to the training data and hardware. The training data consists of 3 specific categories, 1200 images were pictures of Guinea Pigs, 440 were of Chipmunks, and 1000 were of cars. This is less than ideal, because it over-fits the program to these 3 specific categories. A greater diversity of data and categories here would solve this issue. 

The second issue is my hardware,  which faced memory limitations and slow speeds. The memory limitation meant that my batch sizes and the depth of the neural network were limited to small sizes. The batch size should not be too large, but it should not be too small either. In this case the batch sizes were relatively small, meaning that it did not get too much data to train on before going to the next iteration. One of the advantages of batches are to update the weights more frequently when one iteration is going through millions of images. This scenario is the opposite, where small batches were used to fit the memory limitations. The depth of the neural network is also limited. An advantage of deeper neural networks is that it gives the algorithm a greater opportunity to extract more features and higher level features from the data that is then used to classify the image.  

The hardware also limited the speed of my system. This meant the time it took for one iteration was impractically long. It was impractical since training for a many more iterations would mean doing no other work for days and that trade off cost was not worth it since the goal of this project could be achieved anyways. Either way, the main source of the un-generalizability comes from the small un-diverse data set, so training for days on end would have just over fit the model even more. Therefore, more training would have not significantly improved the generalizability of the model anyways. Nonetheless, if the computer was stronger and had a larger and more diverse dataset, then more iterations/epochs would have proven beneficial. I will be upgrading my system soon, so this hardware issue is temporary.

Overall, the fact that it went from classifying 40% of the data to over 80% is a good proof of concept. With the issues mentioned addressed then I am confident that the cross validation set will also reach similar levels. 

