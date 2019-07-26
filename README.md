# 2019-big-data-project-sparkles
2019-big-data-project-sparkles created by GitHub Classroom


# Google Object Detection with Spark Machine Learning
### Yoo Na Cha, Nupur Neti, Michael Schweizer

## Executive Summary


## Introduction 


## Methods 

`How you cleaned, prepared the dataset with samples of intermediate data
Tools you used for analyzing the dataset and the justification (tools, models, etc.)
How did you model the dataset, what techniques did you use and why?
Did you have a hypothesis that you were trying to prove?
Did you just visualize the dataset, and if so, why?`

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
[Insert picture of transfer learning from slides]
The pre-trained model that we used is called InceptionV3. InceptionV3 is an incredibly deep convolutional neural network with dozens of layers. The layers consist of convolution and pooling functions that extract features from image data. On top of these layers of convolutions and pooling functions, there is a fully connected neural network that does the actual classification. 
[Insert picture of InceptionV3 architecture]
InceptionV3 was trained on ImageNet data, which consists of thousands of pictures covering hundreds of object classes. Transfer learning allows us to leverage this incredible model without having to train it ourselves. However, in order to create a model that can classify images in as cars or trees, we need to add our own classifier on top of the dozens of layers of convolution and pooling functions that perform feature extraction. For our classifier, we trained a penalized logistic regression.

#### ML pipeline
Our ML pipeline in Spark consists of three stages: stringIndexer, DeepFeaturizer, and LogisticRegression. stringIndexer converts our labels from strings to numerics. This is necessary because ML pipelines in Spark only allow numeric. DeepFeaturizer is a function from the sparkdl package. It allows to implement transfer learning in Spark. DeepFeaturizer removes the last three layers (the classification layers) of the pre-trained InceptionV3 model. This allows us to train our own classifier that is suited for our task instead. As the classifier, we used LogisticRegression. We added regularization to the LogisticRegression to avoid overfitting.
[At some point, add image of the pipeline from the presentation]



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
