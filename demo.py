import tensorflow as tf
import pandas as pd
import cv2 as cv
import numpy as np
import math
from model import Model

data_dir = 'data'
truth_file = 'truth.csv'

model = Model().get_model()

tf.logging.set_verbosity(tf.logging.DEBUG)

demo_data = pd.read_csv(f'{data_dir}/{truth_file}')
images = np.zeros((0, 400, 400, 3))

for i, row in demo_data.iterrows():
    img = cv.imread(f'{data_dir}/{row["ID"]}.jpg', cv.IMREAD_COLOR)
    img = cv.copyMakeBorder(img, int((400-np.shape(img)[0])/2), math.ceil((400-np.shape(img)[0])/2), int((400-np.shape(img)[1])/2), math.ceil((400-np.shape(img)[1])/2), cv.BORDER_CONSTANT, 0)
    img = np.reshape(img, (1, 400, 400, 3))
    images = np.concatenate((images, img), axis=0)

generator = model.predict(
    input_fn=tf.estimator.inputs.numpy_input_fn(images, shuffle=False, batch_size=6),
)

output = []

for i in range(6):
    prediction = int(round(next(generator)['dense'][0]))
    print('Predicted:', prediction, 'Actual:', demo_data['Class'][i])
    cv.imshow('Image', np.reshape(images[i].astype(np.uint8), (400, 400, 3)))
    cv.waitKey()
