"""
2020 Frc Infinite Recharge
Ball Intake Detection
uses contour lines, rough area calculations
width/height ratios, and radius of contours found
in masked image to find ball
"""

import cv2
import tensorflow as tf
from processing import colors
from processing import cvfilters
from processing import shape_util

debug = False

def predict(img, interpreter, input_details, output_details):

  prediction_data = []

  floating_model = input_details[0]['dtype'] == np.float32
  # Test the model on random input data.
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  image_np = np.asarray(img)
  r_img = cv2.resize(image_np, dsize=(height, width))
  input_data = np.expand_dims(r_img, axis=0)
  if floating_model:
      print("floating")
      input_data = (np.float32(input_data) - 127.5) / 127.5
      print(input_data.shape)
      
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  output_data = interpreter.get_tensor(output_details[0]['index'])
  print(interpreter.get_tensor(output_details[2]['index']))
  print(interpreter.get_tensor(output_details[1]['index']))
  results = np.squeeze(output_data)

  return results
