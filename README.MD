## FoodVision 101
<p align="center">
  <img src="Images/foodapp.jpeg">
</p> <br>

This project is a personal project to develop a computer vision classification model for food images. This is an area of active research (https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf). Thankfully, this means that the FoodVision data is availale as a tensorflow dataset object (To get an overview of all datasets available in tensorflow datasets, go to https://www.tensorflow.org/datasets/overview). There are altogether 101 different classes, some of which look similar - for e.g. Baklava and Apple Pie. We aim to beat the state-of-the-art classification accuracy performance of 77%!

## The approach - using transfer learning
1. Preprocess data by ensuring that all images are scaled to the same dimensions with normalized pixel values.
2. Develop data loader to feed for model training.
3. Experiment with different pre-trained image classification models from Tensorflow as feature extractors. 
4. Unfreeze 5-10 convolution layers in the best performing feature extractor model and finetune the model for 5 more epochs.
5. Model evaluation - ascertain the classes that the model struggles to understand.
6. Quantization aware training - we need lean models for deployment with low latency.
<br>

## To instantiate environment
```
git clone https://github.com/tituslhy/NER
pip -r requirements.txt
```
<br>

## Using the model to run inferences
The flags of the python3 run are:
1. --img: This is the path of the image and is a required argument
2. --pplot: Boolean. Defaults to False. If True, the script plots the image and its predicted classifcation
```
python3 main.py --img "img.jpeg" --plot "True"
```