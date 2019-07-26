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

All our data was sourced from the [Open Images Dataset](https://storage.googleapis.com/openimages/web/download.html) s3 and Google Cloud Storage buckets 
We had two types of data:
* Label information csv files 
* Image jpg files 

#### Label information 
The label csv was stored in a [Google Cloud storage bucket](https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv) and was read in using `pandas read.csv` and then saved as a Spark DataFrame which had 256707 rows and the following schema: 

![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/label_schema.png)

Each image could have multiple labels since a single picture could have multiple objects in it. The labels are alphanumeric codes, which were later joined with the interpretable label names to look like this:

![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/label_joined.png)

The source `verification` implies that the labels were manually verified. 
A confidence of 1 is a positive label and a confidence of 0 is a negative label which tells us that that we can be reasonably sure that the label is NOT in the picture.

For the scope of this project, we decided to keep data with positive labels corresponding to `car` and `tree` 

#### Image files
	
The images were read in from an [s3 bucket](s3://open-images-dataset/validation) using `sparkdl`’s `imageIO` function and saved as a Spark Dataframe with 41620 rows and the following schema:

![](https://github.com/gwu-bigdata/2019-big-data-project-sparkles/blob/master/image_schema.png)

o	`origin` : This was a string containing the s3 link to the image 
o	`height` and `width`: This gave us the dimensions of the image in pixels
o	`nChannels` : This gives us the number of color channels, which was 3 for all the pictures 
o	`mode` : This was `RGB` for our images since they had 3 channels
o	`data` : This contained all of the pixel information in the image as a binary variable

In order to join the data with the label csv, we needed an image ID variable, which was created by extracting only the id from the `origin` string of each image.

The data was joined together to look like this :
[Insert image]



### Explanatory Data Analysis

### Modelling

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
