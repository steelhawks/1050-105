"""
2021 Frc Infinite Recharge
General ML Prediction (Four classes)
uses tflite on transfer learning 
trained MobileNetv2 model (640*640 FPN) 
to detect powercells, loading bay, low 
port, and high port.
"""

import cv2
import numpy as np
import tensorflow as tf

debug = False

def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def detect_objects(interpreter):
  """Returns a list of detection results, each a dictionary of object info."""

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))
  results = []
  for i in range(count):
    result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
    }
    results.append(result)
  return results

def predict(img, interpreter, input_details, output_details):

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
  output_data = detect_objects(interpreter)

  return output_data

def draw(img, output_data):

  red = (255, 0, 0)
  green = (0, 255, 0)
  blue = (0, 0, 255)
  purple = (255, 0, 255)

  for i in range(len(output_data)):
      item = output_data[i]
      if(item['class_id'] == 1 and item['score'] > 0.1):
        image_np = cv2.rectangle(image_np, (int(item['bounding_box'][1] * 640), int(item['bounding_box'][0] * 480)), (int(item['bounding_box'][3] * 640), int(item['bounding_box'][2] * 480)), red, 2)
      if(item['class_id'] == 2 and item['score'] > 0.02):
        image_np = cv2.rectangle(image_np, (int(item['bounding_box'][1] * 640), int(item['bounding_box'][0] * 480)), (int(item['bounding_box'][3] * 640), int(item['bounding_box'][2] * 480)), blue, 2)
      if(item['class_id'] == 3 and item['score'] > 0.1):
        image_np = cv2.rectangle(image_np, (int(item['bounding_box'][1] * 640), int(item['bounding_box'][0] * 480)), (int(item['bounding_box'][3] * 640), int(item['bounding_box'][2] * 480)), green, 2)
      if(item['class_id'] == 4 and item['score'] > 0.1):
        image_np = cv2.rectangle(image_np, (int(item['bounding_box'][1] * 640), int(item['bounding_box'][0] * 480)), (int(item['bounding_box'][3] * 640), int(item['bounding_box'][2] * 480)), purple, 2)

