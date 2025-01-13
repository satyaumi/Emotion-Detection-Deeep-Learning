import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array ,load_img
import gradio as gr

from PIL import Image
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model =tf.keras.models.load_model('Resnet_model_version_5.keras')

emotion_labels ={'angry':0, 'disgust':1,'fear':2,'happy':3,'neutral':4,'sad':5,'surprise':6}
index_to_emotion ={v:k for k,v in emotion_labels.items()}




def prepare_image(image_pil):
  """Preprocess the PIL image to fit your model's input reqirements."""
  # Convert the PIL image to a numpy array with the target size
  img =image_pil.resize((224,224))
  img_array= img_to_array(img)
  img_array =np.expand_dims(img_array,axis=0)
  img_array /=255.0
  return img_array

#define gradio interface
def predict_emotion(image):
    # Preprocess the image
    processed_image =prepare_image(image)
    # Make prediction using the model
    prediction =model.predict(processed_image)
    # Get the emotion label with the highest probability
    predicted_class = np.argmax(prediction, axis=1)
    predicted_emotion = index_to_emotion.get(predicted_class[0], "Unknown Emotion")
    return predicted_emotion


interface =gr.Interface(
    fn=predict_emotion,
    inputs =gr.Image(type='pil'),
    outputs="text",
    title='Emotion Detection',
    description='Upload an image and see the predicted emotion.'
    
)
interface.launch()
