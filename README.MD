## FoodVision 101
<p align="center">
  <img src="Images/foodapp.jpeg">
</p> <br>

This project is a personal project to develop a computer vision classification model for food images. This is an area of active research (https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf). Thankfully, this means that the FoodVision data is availale as a tensorflow dataset object (To get an overview of all datasets available in tensorflow datasets, go to https://www.tensorflow.org/datasets/overview). There are altogether 101 different classes, some of which look similar - for e.g. Baklava and Apple Pie. We aim to beat the state-of-the-art classification accuracy performance of 77%.

<br><br><br>



## The approach - using transfer learning
1. Preprocess data by ensuring that all images are scaled to the same dimensions with normalized pixel values.
2. Develop data loader to feed for model training.
3. Experiment with different pre-trained image classification models from Tensorflow as feature extractors. 
4. Unfreeze 5-10 convolution layers in the best performing feature extractor model and finetune the model for 5 more epochs.
5. Model evaluation - ascertain the classes that the model struggles to understand.
6. Quantization aware training - we need lean models for deployment with low latency. This is a bonus objective, as the best model might not adopt mixed precision training.
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

## Notes
1. There is a bug when using mixed precision training with EfficientNet. The current fix is to use Tensorflow version 2.4 instead.
2. EfficientNet models have an input normalization in-built, so there is no need to scale image pixels between 0-255 during  preprocessing.
3. Model training and annotations are in under `Notebooks > FoodVision101 Model Development.ipynb`

## <h3> Experimental results and findings </h3>
We experimented with the following feature extractors before finetuning: <br>
<table>
  <thead align="center">
    <tr border: none;>
      <td><b>Experiment</b></td>
      <td><b>Feature extractor</b></td>
      <td><b>Validation accuracy (on 15% of test data)</b></td>
      <td><b>Findings</b></td>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td><b>InceptionResNetV2 with Data Augmentation</b></a></td>
        <td>InceptionResNetV2: The InceptionResNetV2 model applies residual connections to Google's in-house InceptionV3 model. https://arxiv.org/pdf/1602.07261.pdf</td>
        <td>55%</td>
        <td>The use of data augmentation causes the model to severely underfit instead of helping it generalize better.</td>
    </tr>
    <tr>
        <td><b>InceptionResNetV2</b></a></td>
        <td>InceptionResNetV2: The InceptionResNetV2 model applies residual connections to Google's in-house InceptionV3 model. https://arxiv.org/pdf/1602.07261.pdf</td>
        <td>61%</td>
        <td>Performance improved significantly, but it's still not great. This could be due to choice of feature extractor model</td>
    </tr>
    <tr>
        <td><b>EfficientNetB0</b></a></td>
        <td>EfficientNetB0: EfficientNet boasts very high accuracy despite very few parameters. The backbone of EfficientNet models is similar to MobileNetV2 in that it uses mobile inverted bottleneck convolution. Where it differs from other models is the 'efficiency' when scaling width, depth, resolution or a scaling combination. (https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)</td>
        <td>75%</td>
        <td>Performance improved significantly with much faster training times than prior experiments.</td>
    </tr>
    <tr>
        <td><b>EfficientNetB4</b></a></td>
        <td>EfficientNetB4: This is a larger version of EfficientNetB0. The thinking behind this experiment is: if some is good, more must be better.</td>
        <td>72%</td>
        <td>Performance is slightly lower with much slower training times.</td>
    </tr>
    <tr>
        <td><b>EfficientNetV2B0</b></a></td>
        <td>EfficientNetV2B0: EfficientNetV2 uses the concept of progressive learning. This means that although the image sizes are originally small when the training starts, they increase in size progressively. Scaling and regularization are done dynamically throughout model training. (https://towardsdatascience.com/google-releases-efficientnetv2-a-smaller-faster-and-better-efficientnet-673a77bdd43c)
        </td>
        <td>75.5%</td>
        <td>Performance for this experiment is the best. We therefore choose to finetune this experiment further</td>
    </tr>
    <tr>
        <td><b>EfficientNetV2B0 Finetuned</b></a></td>
        <td>EfficientNetV2B0 with all layers set to trainable.
        </td>
        <td>84.6%</td>
        <td>We successfully exceed the target we set ourselves!</td>
    </tr>