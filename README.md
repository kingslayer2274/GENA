## About Us
We are a group of senior students in faculty of Engineering, and we choosed to make a
 “Brain Tumor Radio Genomic Classification” using deep learning as a graduation project .

## What We Are Working On
We’re tackling the problem of surgical intervention to know whether a brain tumor will have a specific genetic sequence
(MGMT promoter methylation) or not

## Procedures
1. Develop a system which help in predicting the genetics of the cancer through imaging
2. The model will be covered by an interface that will do the image processing required before deciding
3. A developer wants to access our ML model API, he can submit a request via the website and access the API to use it

## CURRENT SOLUTIONS

Currently, genetic analysis of cancer requires surgery to extract a tissue sample.

![Screenshot 2022-07-02 221853](https://user-images.githubusercontent.com/57845488/177015076-565988bd-dbe9-45a5-b7fc-96a395852400.png)

## Project Stages
![Screenshot 2022-07-02 222119](https://user-images.githubusercontent.com/57845488/177015124-83315491-7016-43d8-bd32-7f96dbaa9796.png)

we collected data from : **RSNA**

the structure of MRI Scans was like this :

![Screenshot 2022-07-02 222658](https://user-images.githubusercontent.com/57845488/177015289-c2e18eb0-a154-4fb7-9ad3-96a929e4555d.png)

the data has **585** patient each patient has 4 kind of scans **["FLAIR" , "T1w" , "T1wCE" , "T2w"]**

## Central image trick:
Each independent case has a different number of images for all the MRI scans. Using all the scans will confuse the model to learn the spatial dependence of the brain pixels.
For example, Let’s assume that case_1 has 80 T1wCE scans and case_2 has 500 T1wCE scans and that we will use 40 images as an input for the model.
In this example, we will end up with 2 3d images that don’t represent the same portion of the brain. Thus, the model will not only fail to learn the tumor pattern but will also start learning some spatial patterns that are not useful in our case.
What we wanted to do and failed is to select a fixed starting and ending point for all the brains and train with the same information for all the cases.
we found that using the biggest image as a central image (the image that contains the largest brain cutaway view). 
We think that this was the only 100% successful experiment We did in this competition.


https://github.com/helloflask/flask-examples/tree/master/template
