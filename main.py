import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# Instantiate parser
parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument('--img',
                    type  = str,
                    required = True,
                    help = "Path to directory containing image. Defaults to 'test_image'")
parser.add_argument('--plot',
                    type = bool,
                    required = False,
                    help = "Whether to plot the image after classifying. Defaults to False")
parser.add_argument('--q',
                    type = bool,
                    required = False,
                    help = "Whether to use quantized models to run inference. Defaults to False")

args = parser.parse_args()

# Instantiate environment variables
IMG_PATH = args.img_path

if args.plot:
    PLOT = args.plot
else:
    PLOT = False

if args.q:
    QUANTIZE = args.q
else:
    QUANTIZE = False

if QUANTIZE == True:
    model_path = './Model/best_model_q.h5'
    weight_path = './Model/best_weights_q.h5'
else:
    model_path = './Model/best_model.h5',
    weight_path = './Model/best_weights.h5'

labels_list = open('file.txt','r').read().split('\n')
labels = {num:name for num,name in enumerate(labels_list)}

def run_inference(img_path = IMG_PATH,
                  plot = PLOT,
                  model_path = model_path,
                  weight_path = model_path,
                  ):
    """_summary_
    Runs inference on the given image.

    Args:
        img_path (str, required): Path to image to run inference.
        plot (bool, optional): Option to plot image with label. Defaults to PLOT.
        model_path (str): Defaults to 'best_model.h5'.
        weight_path (str): Defaults to 'best_weights.h5'.

    Returns:
        label (str): Returns the model's predicted label for the image.
    """
    model = tf.keras.models.load_model(model_path, compile = False)
    model.load_weights(weight_path)
    pixels = np.expand_dims(np.asarray(Image.open(img_path).resize((224,224))).astype('float32'),0)
    pixels /= 255.
    label = labels[np.argmax(model.predict(pixels))]
    
    if plot==True:
        plt.imshow(np.squeeze(pixels))
        plt.title(f'This image is labelled as: {label}. Did we get it right?')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
    
    return label

if __name__ == '__main__':
    run_inference(IMG_PATH, PLOT)