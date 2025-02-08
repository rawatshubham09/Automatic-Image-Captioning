# Automatic-Image-Captioning
"CDAC Final Project

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml
10. app.py

## data validation pipeline

1. create a function which check: 
    - jpg formate
    - size of image 
    - image caption in csv file
2. split the image validation and train data
    - to save space I am going to 

## model building pipeline
1. download densenet201 and convert image into vector forms
2. create a model building python file and Custome ImageDataGenerator
3. train model with train and test data