"""
This script is designed to create a decision system for polyp detection using various libraries including Streamlit for web app development, TensorFlow for deep learning models, OpenCV for image processing, and Firebase for database management. The application allows users to upload endoscopic images, which are then processed to detect polyps. Users can sign up, log in, and save medical records associated with the images they upload. The application also features a functionality to generate and download a report in PDF format.
"""

# Import necessary libraries
from cProfile import label  # Used for profiling (consider removing if not used)
from re import sub  # Regular expression operations (consider removing if not used)
from tkinter.tix import CELL  # Tkinter GUI toolkit (consider removing if not used)
import streamlit as st  # Streamlit library for creating web applications
# import streamlit_authenticator as stauth  # Uncomment if using Streamlit authenticator
import pyrebase  # Pyrebase for Firebase interaction
from google.cloud import firestore  # Firestore for database operations
from fpdf import FPDF  # FPDF for PDF generation
import base64  # Base64 for encoding
import numpy as np  # NumPy for numerical operations
import cv2  # OpenCV for image processing
from tensorflow.keras.utils import CustomObjectScope  # TensorFlow for model customization
from tqdm import tqdm  # tqdm for progress bars (consider removing if not used)
from tensorflow.keras import backend as K  # TensorFlow backend
import os  # OS module for interacting with the operating system
import tensorflow as tf  # TensorFlow for machine learning models
from pdfrw import PageMerge, PdfReader, PdfWriter  # PDFrw for PDF manipulation
from datetime import datetime, date, time  # DateTime for handling dates and times
from streamlit import session_state  # Streamlit session state for maintaining session data
from PIL import Image, ImageDraw  # PIL for image processing
import io  # IO for input/output operations
from io import BytesIO  # BytesIO for byte stream operations
import collections  # Collections for specialized container datatypes
import time  # Time for time-related tasks
import random  # Random for generating random numbers
try:
    from collections import abc
    collections.MutableMapping = abc.MutableMapping  # Patch for collections MutableMapping
except:
    pass

# Firebase configuration (Consider moving sensitive data to a secure location)
firebaseConfig = {
    # Your Firebase configuration goes here
}

# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Database
db = firebase.database()
storage = firebase.storage()
st.sidebar.title("Decision System For Polyp Detection")

# Authentication
choice = st.sidebar.selectbox('login/Signup', ['Login', 'Sign up'])

# Obtain User Input for email and password
email = st.sidebar.text_input('Please enter your email address')
password = st.sidebar.text_input('Please enter your password',type = 'password')

# Function to calculate Intersection over Union (IoU)
def iou(y_true, y_pred):
    # Function to calculate IoU for a single prediction
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

# Dice coefficient for evaluating segmentation models
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# Dice loss function
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# Read and preprocess image
def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x

# Read and preprocess mask
def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x

# Parse mask for visualization
def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2,
