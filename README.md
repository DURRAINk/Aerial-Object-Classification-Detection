# 🛩️ Aerial Object Classification & Detection
streamlit app: https://aerial-object-classification-and-detection.streamlit.app/

This project provides a deep learning-based solution to classify aerial images into two categories — **Bird** or **Drone** — and optionally perform **object detection** to locate and label these objects in real-world scenes.

The solution is designed for applications in:
- **Security surveillance**
- **Wildlife protection**
- **Airspace safety**

Accurate identification between drones and birds is critical in these domains.

---

## 🚀 Features
- **Custom CNN Model** for aerial image classification
- **Transfer Learning** with pretrained models for improved accuracy
- **YOLOv8 Integration (Optional)** for real-time object detection
- **Streamlit Deployment** for interactive web-based usage
- **Modular Codebase** with clear separation of training, inference, and deployment

---

## 📂 Project Structure
    Aerial-Object-Classification-Detection/ 
    │── models/ 
    │    └── bird_drone_classifier.h5  # Classification model
    |    └── bird_drone.pt  # yolo detection model
    └── Aerial.ipynb # Jupyter notebook for training & experiments 
    │── app.py # Streamlit app for deployment 
    │── requirements.txt # Python libraries 
    │── packages.txt # System-level packages (if needed) 
    │── README.md # Project documentation


---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DURRAINk/Aerial-Object-Classification-Detection.git
   cd Aerial-Object-Classification-Detection

2. Install dependencies/libraries:
   ```bash
   pip install -r requirements.txt

3. (Optional) Install system packages:
   ```bash
   xargs -a packages.txt sudo apt-get install -y

## To run streamlit app
  
     streamlit run app.py

## Thankyou!
