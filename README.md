2019-big-data-project-sparkles created by GitHub Classroom
# Image Classification with Spark Machine Learning
2019-big-data-project-sparkles

Yoo Na Cha, Nupur Neti, Michael Schweizer


## Executive Summary

Through this project we were able to:

1. Gain exposure to reading in image datasets into a distributed file system
2. Learn how to configure Spark clusters to add any necessary libraries
3. Practice processing very large unstructured datasets using Spark
4. Become familiar with conducting machine learning in Spark using `mllib` and `sparkdl`


| Navigation |
|---|
| [Introduction](README.md#Introduction) |
| [Analytical Methods](README.md#Analytical-Methods)|
| [Results & Conclusions](README.md#Results-and-Conclusions) | 
| [Future Work](README.md#Future-Work)


| Code Files |
|---|
| [Image Classification Notebook](image-classification-notebook.ipynb)|


## Introduction 

In previous projects, we have encountered problems where the computer memory did not have enough capacity to execute the models or took excessively long to do so. For image classification projects, where larger datasets and expensive computations are required, this issue is especially common. 

In regards, we chose `image classification` as the topic for this final project to explore how we could apply what we learned in this class to overcome such limitations. 

### Dataset
We used the dataset from the [Open Images 2019 - Object Detection](https://www.kaggle.com/c/open-images-2019-object-detection) competition from Kaggle. The dataset provides a large quantitiy of images which are each annotated with labels, indicating certain object classes are present within a particular image. Due to restraints on the budget and time, we decided to use the validation dataset for our project, which had 193,300 image-level labels and is 12GB in total size. 

## Analytical Methods 

### Tools
For this project, we have used Spark on AWS EMR. We have conducted all project steps in Spark, including data sourcing and ingesting, exploratory data analysis, modeling, and evaluation of results. 

#### Software
Because we are all Python programmers, we used the Python API for Spark, called PySpark, to conduct our project. In PySpark, we used the structured APIs Dataframes and SQL. In terms of packages, we used the `mllib` package and the `sparkdl` package. [Sparkdl](https://github.com/databricks/spark-deep-learning) is a package for Deep Learning in Spark and allowed us to make use of Transfer Learning when creating our classification model. In order to run `sparkdl`, we also had to install `Tensorflow` and `Keras` on all the machines in our cluster. Because of the complicated setup with `sparkdl`, `Tensorflow`, and `Keras`, we made use of a special bootstrap script when launching our cluster. You can find the bootstrap script here: s3://bigdatateaching/bootstrap/bigdata-deeplearning-bootstrap.sh

#### Cluster hardware
When conducting our project, we had to go through several iterations to figure out the best hardware setup. We started out with 6 m4.xlarge instances (1 master and 5 workers) with 16GB of memory each. This setup worked fine when we tested our code on a small subset of the data. However, when we ran it on the entire dataset, we ran into memory errors. The cluster did not have enough memory to train our deep learning model. To combat this, we increased the instance types to m4.2xlarge instances with 32GB of memory. However, we ran into memory issues again. As a next step, we increased the instances to m4.16xlarge with 256GB of memory. But, we ran into memory issues once again. Therefore, we decided to use the largest instances available on AWS: r5.24xlarge. This instance type has 768GB of memory and costs $6 per hour. We ran a cluster of 1 master and 3 workers and finally managed to train and evaluate our model. 

### Data Sourcing and Ingesting

All our data was sourced from the [Open Images Dataset](https://storage.googleapis.com/openimages/web/download.html) s3 and Google Cloud Storage buckets.

We had two types of data:
* Label information csv files 
* Image jpg files 

#### Label information 
The label csv was stored in a [Google Cloud storage bucket](https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv) and was read in using `pandas read.csv` and then saved as a Spark DataFrame which had 256707 rows and the following schema: 

![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/data/label_schema.png)

Each image could have multiple labels since a single picture could have multiple objects in it. The labels are alphanumeric codes, which were later joined with the interpretable label names to look like this:

![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/data/label_joined.png)

* `Source` value of  `verification` implies that the labels were manually verified. 
* A `Confidence` of 1 is a positive label and a confidence of 0 is a negative label which tells us that that we can be reasonably sure that the label is NOT in the picture.

For the scope of this project, we decided to keep data with positive labels corresponding to `car` and `tree` 

#### Image files
	
The images were read in from an [s3 bucket](s3://open-images-dataset/validation) using `sparkdl`’s `imageIO` function and saved as a Spark Dataframe with 41620 rows and the following schema:

![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/data/image_schema.png)

* `origin` : This was a string containing the s3 link t!o the image 
* `height` and `width`: This gave us the dimensions of the image in pixels
* `nChannels` : This gives us the number of color channels, which was 3 for all the pictures 
* `mode` : This was `RGB` for our images since they had 3 channels
* `data` : This contained all of the pixel information in the image as a binary variable

In order to join the data with the label csv, we needed an image ID variable, which was created by extracting only the id from the `origin` string of each image.

The data was joined together to look like this :

![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/data/image_joined.png)

### Explanatory Data Analysis

We have initially used sparkdl's `imageIO` to load images from S3 bucket. From the data, we were able to check the dimension of each images, which are as follows: 

![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/data/image-detection.PNG)

However, because the original schema of the files made it hard for explanatory data analysis, just for this section, we downloaded the validation dataset and read it into Spark. 

Firstly from importing the image label dataset, we were able to see that 41620 images were matched with 256,707 labels. This indicates that each image had several object labels within. After leaving just the necessary columns, Image ID and LabelName, we joined the dataset with actual names of the labels for better interpretability. Through the `.count()` function, we were able to see that there were 601 unique label names in the dataset. 

With the preprocessed dataset containing image ID and its matching label names, we went to do some explanatory data analysis to better understand the data. 

As mentioned above, we could see that multiple labels were included in one image file, varying from 1 to 25 labels. The distribution of label counts in one image was right skewed with most of the images having around five labels. 

![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/data/label-count.png)

When comparing the labels with the actual label names, we were able to ses that some of the most frequent label names were `mammal`, `person`, `plant`, or `clothing`. 

![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/data/label-frequency.png)

More interestingly, we created a dendogram of the label names which shows which labels were likely to appear together within one image. With the top 48 images, the dendogram hierarchically clusters labels that have high correlation in occurances. We can see that human eye, nose or any other parts of the face were very highly correlated.

![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/data/label-clustering.PNG)


### Modeling
#### Transfer learning approach
For our actual ML model, we used the transfer learning approach. Transfer learning means that we take a pre-trained model and re-train it so that it fits our dataset.
![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/data/transferlearning.png)
The pre-trained model that we used is in this project is InceptionV3. InceptionV3 is an incredibly deep convolutional neural network with dozens of layers. The layers consist of convolution and pooling functions that extract features from image data. On top of these layers of convolution and pooling functions, there is a fully connected neural network that does the actual classification. Here is an illustration of InceptionV3's network architecture:
![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/data/inceptionV3.png)
InceptionV3 was trained on ImageNet data, which consists of thousands of pictures covering hundreds of object classes. Transfer learning allows us to leverage this incredible model without having to train it ourselves. However, in order to create a model that can classify images as cars or trees, which was the objective of our model, we need to add our own classifier on top of the dozens of layers of convolution and pooling functions that perform feature extraction. For our classifier, we trained a penalized logistic regression.

#### ML pipeline
Our ML pipeline in Spark consists of three stages: `stringIndexer`, `DeepFeaturizer`, and `LogisticRegression`. stringIndexer converts our labels from strings to numerics. This is necessary because ML pipelines in Spark only allows numeric inputs. DeepFeaturizer is a function from the sparkdl package. It allows to implement transfer learning in Spark. DeepFeaturizer removes the last three layers (the classification layers) of the pre-trained InceptionV3 model. This allows us to train our own classifier that is suited for our task instead. As the classifier, we used LogisticRegression. We added regularization to the LogisticRegression to avoid overfitting. Here is an illustration of our final pipeline:
![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/data/mllibpipeline.png)



## Results and Conclusions

Since our problem was that of binary classification (cars vs. trees), we measured our model on the following metrics:

* Accuracy : 0.9265
* F1 Score : 0.9266
* AUC : 0.9286

Our F1 Score and AUC give us a sense that our model is not performing differently while predicting between cars and trees, which is good.

Considering the fact that the original Kaggle problem was a classification problem with 600 classes, our accuracy on a relatively smaller binary classification problem could be better. However, an accuracy of 0.9265 can be considered an understimated value since our accuracy takes a hit for every picture that has both car and trees in it.

![Example of an images that we classified wrong](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/data/wrong_classification_1.png) 

We will discuss some ideas to improve our model in the Future Work section. 

## Challenges 

A few aspects of this project that we found challenging were : 
* Unreliability of Spark Cluster
* Setting up the libraries on the cluster
* Debugging Spark and understanding explain paths
* Running out of RAM for joining and training 

## Future Work
Based on the learnings from this project, here are some things that we would do differently and/or expand on:
- Explore the use of AWS SageMaker and/or Azure Machine Learning
- Train and evaluate the model on instances with GPUs
- Understand explain paths better to write more efficient PySpark code
- Train a neural network as the classifier, instead of a logistic regression
- Conduct a grid search to find the best hyperparameters for the classifier
