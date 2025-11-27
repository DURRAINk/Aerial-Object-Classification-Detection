import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from ultralytics import YOLO

#loading pre-trained models (dummy paths for illustration)
@st.cache_resource
def load_models():
    cls = tf.keras.models.load_model('./models/bird_drone_classifier.h5')
    detc = YOLO('./models/bird_drone.pt')
    return cls, detc
classif_model, detct_model = load_models()

def classification(img):
    classes = ['Bird', 'Drone']
    img = load_img(img, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = tf.expand_dims(img, axis=0)
    pred = classif_model.predict(img)[0][0]
    conf =  pred if pred >= 0.5 else 1 - pred
    return classes[round(pred)] , conf

def detection(img):
    img = load_img(img)
    results = detct_model.predict(img, conf=0.25 ,save=False,save_txt=False,save_conf=False,device='cpu')
    pred_img = results[0].plot()
    print(results)
    return pred_img


st.title("Aerial Object Classification & Detection")
sidbr = st.sidebar
sidbr.title("Navigation")
option = sidbr.radio("Go to", ["Image Classification", "Object Detection"])

## For Image Classification ----------
if option == "Image Classification":
    st.header("Image Classification")
    st.write("A deep learning-based solution that can classify aerial images into two categories — Bird or Drone")

    image = st.file_uploader("Upload an aerial image", type=["jpg", "jpeg", "png"])
    
    #img_array = np.expand_dims(img_array, axis=0) 
    if image is not None:
        st.image(image, caption='Uploaded Image', use_container_width=True)
        if st.button("Classify Image"):
            classification_result, conf = classification(image)
            st.success(f'{classification_result} detected with {conf*100:.2f}% confidence')

## For Object Detection ----------
elif option == "Object Detection":
    st.header("Object Detection")
    st.write("A deep learning-based solution that can detect and localize birds and drones in aerial images")

    image = st.file_uploader("Upload an aerial image", type=["jpg", "jpeg", "png"])
    if image is not None:
        if st.button("Detect Objects"):
            pred_img = detection(image)
            st.image(pred_img, caption='Detected Objects', use_container_width=True)
            st.success("Detection Completed")        