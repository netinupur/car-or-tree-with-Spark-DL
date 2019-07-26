2019-big-data-project-sparkles created by GitHub Classroom
# Image Classification with Spark Machine Learning
2019-big-data-project-sparkles

Yoo Na Cha, Nupur Neti, Michael Schweizer


## Executive Summary

Through the project we were able to:

1. Expand experience on reading in image datasets into the distributed system
2. Learn how to configure cluster configurations to add any necessary libraries
3. Overcome the limitations on processing large dataset using Spark and compare the advantages/disadvantages with modelling without distributed system
4. Become familiar with deep learning process using `mllib`

## Introduction 

In previous projects, we have encountered problems where the computer memory did not have enough capacity to execute the models or took excessively long amount of time. Especially with projects on image classification, where larger dataset and expensive computation is required, the issue was worse. 

In regards, we chose `image classification` as the topic for the final project and see how we could apply what we learned in this class to overcome such limitations. 

#### Dataset
We used dataset from [Open Images 2019 - Object Detection](https://www.kaggle.com/c/open-images-2019-object-detection) competition from Kaggle. The dataset provides large amount of image files which are each annotated with labels, indicating certain object classes are present within the image. Due to restraints on the budget and time, we decided to use the validation dataset for the project, which had 193,300 image-level labels and 12GB in total size. 

## Methods 

### Tools
For this project, we have used Spark on AWS EMR. We have conducted all project steps in Spark, including data sourcing and ingesting, exploratory data analysis, modeling, and evaluation of results. 

#### Software
Because we are all Python programmers, we used the Python API for Spark called PySpark to conduct our project. In PySpark, we used the structured APIs Dataframes and SQL. In terms of packages, we used the mllib package and the sparkdl package. [Sparkdl] (https://github.com/databricks/spark-deep-learning) is a package for Deep Learning in Spark and allowed us to make use of Transfer Learning when creating our classification model. In order to run sparkdl, we also had to install Tensorflow and Keras on all the machines in our cluster. Because of the complicated setup with sparkdl, Tensorflow, and Keras, we made use of a special bootstrap script when launching our cluster. You can find the bootstrap script can be found [here] (s3://bigdatateaching/bootstrap/bigdata-deeplearning-bootstrap.sh).

#### Cluster hardware
When conducting our projects, we had to go through several iterations to figure out the best hardware setup. We started out with 6 m4.xlarge instances (1 master and 5 workers) with 16GB of memory each. This setup worked fine when we tested our code on a small subset of our data. However, when we ran it on the entire dataset, we ran into memory errors. The cluster did not have enough memory to train our deep learning model. To combat this, we increased the instance types to m4.2xlarge instances with 32GB of memory. However, we ran into memory issues again. As a next step, we increased the instances to m4.16xlarge with 256GB of memory. But we ran into memory issues again. Therefore, we decided to use the largest instances available on AWS: r5.24xlarge. This instance type has 768GB of memory and costs $6 per hour. We ran a cluster of 1 master and 3 workers and finally managed to train and evaluate our model. 



### Data Sourcing and Ingesting

All our data was sourced from the [Open Images Dataset](https://storage.googleapis.com/openimages/web/download.html) s3 and Google Cloud Storage buckets.

We had two types of data:
* Label information csv files 
* Image jpg files 

#### Label information 
The label csv was stored in a [Google Cloud storage bucket](https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv) and was read in using `pandas read.csv` and then saved as a Spark DataFrame which had 256707 rows and the following schema: 

![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/label_schema.png)

Each image could have multiple labels since a single picture could have multiple objects in it. The labels are alphanumeric codes, which were later joined with the interpretable label names to look like this:

![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/label_joined.png)

* `Source` value of  `verification` implies that the labels were manually verified. 
* A `Confidence` of 1 is a positive label and a confidence of 0 is a negative label which tells us that that we can be reasonably sure that the label is NOT in the picture.

For the scope of this project, we decided to keep data with positive labels corresponding to `car` and `tree` 

#### Image files
	
The images were read in from an [s3 bucket](s3://open-images-dataset/validation) using `sparkdl`’s `imageIO` function and saved as a Spark Dataframe with 41620 rows and the following schema:

![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/image_schema.png)

* `origin` : This was a string containing the s3 link t!o the image 
* `height` and `width`: This gave us the dimensions of the image in pixels
* `nChannels` : This gives us the number of color channels, which was 3 for all the pictures 
* `mode` : This was `RGB` for our images since they had 3 channels
* `data` : This contained all of the pixel information in the image as a binary variable

In order to join the data with the label csv, we needed an image ID variable, which was created by extracting only the id from the `origin` string of each image.

The data was joined together to look like this :

![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/image_joined.png)

### Explanatory Data Analysis

### Modelling
#### Transfer learning approach
For our actual ML model, we used the transfer learning approach. Transfer learning means that you take a pre-trained model and re-train it so that it fits your dataset.
![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/transferlearning.png)
The pre-trained model that we used is called InceptionV3. InceptionV3 is an incredibly deep convolutional neural network with dozens of layers. The layers consist of convolution and pooling functions that extract features from image data. On top of these layers of convolutions and pooling functions, there is a fully connected neural network that does the actual classification. 
![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/inceptionV3.png)
InceptionV3 was trained on ImageNet data, which consists of thousands of pictures covering hundreds of object classes. Transfer learning allows us to leverage this incredible model without having to train it ourselves. However, in order to create a model that can classify images in as cars or trees, we need to add our own classifier on top of the dozens of layers of convolution and pooling functions that perform feature extraction. For our classifier, we trained a penalized logistic regression.

#### ML pipeline
Our ML pipeline in Spark consists of three stages: stringIndexer, DeepFeaturizer, and LogisticRegression. stringIndexer converts our labels from strings to numerics. This is necessary because ML pipelines in Spark only allow numeric. DeepFeaturizer is a function from the sparkdl package. It allows to implement transfer learning in Spark. DeepFeaturizer removes the last three layers (the classification layers) of the pre-trained InceptionV3 model. This allows us to train our own classifier that is suited for our task instead. As the classifier, we used LogisticRegression. We added regularization to the LogisticRegression to avoid overfitting.
![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/mllibpipeline.png)



| Code Files |
|---|
| [0. Data Sourcing and Ingesting] |
| [1. Explanatory Data Analysis](explanatory-data-analysis.ipynb)|
| [2. Modelling] |


## Results/Conclusion
`What did you find and learn?`
`How did you validate your results?`


## Future Work
`what would you do differently and what follow-up work would you do?`
