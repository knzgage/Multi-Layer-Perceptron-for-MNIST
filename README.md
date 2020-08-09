# Multi-Layer-Perceptron-for-MNIST
A MLP built in python to classify the MNIST hand-written numbers data set

<h2>About the Data</h2>
The data that I worked with can be found

[here](https://www.kaggle.com/oddrationale/mnist-in-csv).

The data is split up into 60,000 training samples and 10,000 test samples, all of which have labels.
Each of these examples correspond to a handwritten number, so all examples contain the label and then 784 values corresponding to
the pixel intensity of the original 24x24 images.

<h2>What I Learned</h2>
This was a project for my first machine learning class in Spring 2020, and it was designed so that
we could learn how to work with large sets of data and implement basic machine learning concepts.
I performed three different experiments to see how different hyperparameter can alter the performance
of a multi-layer perceptron. The three hyperparameters tested were: the number of hidden layers,
the momentum value, and the number of training examples. My results for these experiments are described below.

<h3>Varying the Number of Units</h3>
For this experiment, the momentum was fixed at .9 and testing was conducted over 50 epochs.
The number of hidden layers was tested at 20, 50, and 100.<br/><br/>

![20 Hidden Layers](https://user-images.githubusercontent.com/40836138/89721481-d29a4680-d992-11ea-8d69-39106b6705db.png)

![50 Hidden Layers](https://user-images.githubusercontent.com/40836138/89721781-9b2d9900-d996-11ea-8ff8-672e593f2464.png)

![100 Hidden Layers](https://user-images.githubusercontent.com/40836138/89721786-af719600-d996-11ea-8905-d4f84ee41d87.png)

I discovered that as the number of hidden layers increased, the accuracy on both the training and test sets increased.
In addition to accuracy increasing, the number of epochs needed for the accuracy to reach its max decreased.

<h3>Changing the Momentum</h4>
The MLP uses a calculus technique called gradient descent in order to find the weight values that minimize the
error of the testing set. The momentum hyperparameter determines how much the weights are altered after each epoch.
For this experiment, momentum was tested at 0, 0.25, 0.5, and 0.9.<br/><br/>

![Momentum = 0](https://user-images.githubusercontent.com/40836138/89721797-bef0df00-d996-11ea-9dba-b5e8fab6e09d.png)

![Momentum = 0.25](https://user-images.githubusercontent.com/40836138/89721802-cadca100-d996-11ea-9b46-bd2bfcfbe934.png)

![Momentum = 0.50](https://user-images.githubusercontent.com/40836138/89721815-e34cbb80-d996-11ea-810c-e20049f99738.png)

![Momentum = 0.90](https://user-images.githubusercontent.com/40836138/89721820-f2cc0480-d996-11ea-9138-a00ffe1b6904.png)
  
The momentum didn't seem to have much of the impact on the final accuracy of the test data, but I noticed that
as the momentum increased the number of epochs required for the data to converge also increased. This is likely due
to the momentum being so high that it kept going overshooting the global minima. I certain situations, if the momentum was
too low, our weights could get stuck at a local minima but that didn't appear to be the case since all tests had relatively
similar accuracies.

<h3>Altering the Size of the Training Data</h3>
In this final test, I observed how altering the size of the training set impacted the accuracy of the teset set.
I used the full training set, half the training set, and the a quarter of the training set.<br/><br/>

![Full Training Set](https://user-images.githubusercontent.com/40836138/89721829-037c7a80-d997-11ea-9aca-b0971f945e86.png)

![Half Training Set](https://user-images.githubusercontent.com/40836138/89721831-12fbc380-d997-11ea-96e3-171207ce3bdb.png)

![Quarter Training Set](https://user-images.githubusercontent.com/40836138/89721838-2444d000-d997-11ea-932e-223f534341a6.png)

  
As expected, the accuracy of the MLP on the test set decreased as the size of the training set decreased. This is likely
due to the MLP overfitting to the training data. As the size of the training set decreased, my model's predictive power decreased,
this is because the model didn't get to witness as many different ways to write each of the numbers.
