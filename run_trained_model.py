#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 19:50:24 2024
@author: zhanjunfei

X is an array of shape (N, ) where each element is a WAV file path. Y is of shape (N, ), where each element is a class ID integer.

To load the data of X and Y from wav files to wav paths, you can use this code:
    # Define label mappings
    CLASS_TO_LABEL = {
        'water': 0,
        'table': 1,
        'sofa': 2,
        'railing': 3,
        'glass': 4,
        'blackboard': 5,
        'ben': 6,
    }
    LABEL_TO_CLASS = {label: class_name for class_name, label in CLASS_TO_LABEL.items()}
    
    def load_data(base_dir, class_to_label): 
        X, Y = [], []
        for class_name, label in class_to_label.items():
            class_folder = os.path.join(base_dir, class_name)
            if os.path.isdir(class_folder):
                # List all .wav files in the class folder
                wav_files = [
                    os.path.join(class_folder, file)
                    for file in os.listdir(class_folder)
                    if file.endswith('.wav')
                ]
                X.extend(wav_files)
                Y.extend([label] * len(wav_files))
        return np.array(X), np.array(Y)
    
    # Define directories
    dir = ''
    
    # Load training data
    X, Y = load_data(dir, CLASS_TO_LABEL)
"""
#import necessary libaries
import os
import gdown
import zipfile
import numpy as np
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from google.colab import drive


def run_trained_model(X):
    
    def download_model_files():
        """
        Downloads four .npy model files from Google Drive using gdown.
        
        Files:
        - model_bias.npy
        - model_bias_3.npy
        - model_weights.npy
        - model_weights_3.npy
        """
        # List of files with their corresponding Google Drive URLs and desired output names
        files = [
            {
                "url": "https://drive.google.com/file/d/1SkKeIv5wM0DucEIbgwcNfU_1HS5R_nlZ/view?usp=drive_link",
                "output": "/content/model_bias.npy"
            },
            {
                "url": "https://drive.google.com/file/d/1VqpOwiC3aVWNLjnq-JM1LXXDZLQSpl5T/view?usp=drive_link",
                "output": "/content/model_bias_3.npy"
            },
            {
                "url": "https://drive.google.com/file/d/1HYQ4ELQpJ4JQj6ay7eo2n2jiJ4yvkqQ4/view?usp=drive_link",
                "output": "/content/model_weights.npy"
            },
            {
                "url": "https://drive.google.com/file/d/1BNXQX0qflp6Gi0-m1GvCMO7Z8UIHNbIK/view?usp=drive_link",
                "output": "/content/model_weights_3.npy"
            }
        ]
        
        # Directory where files will be downloaded
        download_dir = "model_files"
        os.makedirs(download_dir, exist_ok=True)  # Create the directory if it doesn't exist
        
        for file in files:
            url = file["url"]
            output_path = os.path.join(download_dir, file["output"])
            
            try:
                print(f"Downloading {file['output']}...")
                gdown.download(url, output_path, quiet=False, fuzzy=True)
                print(f"Downloaded {file['output']} successfully.\n")
            except Exception as e:
                print(f"Failed to download {file['output']}. Error: {e}\n")
    
        print("All download attempts completed.")

    # Execute the download function
    download_model_files()
    
    #load the parameters for the classifier
    weights1 = np.load("/content/model_weights.npy")
    weights2 = np.load("/content/model_weights_3.npy")
    bias1 = np.load("/content/model_bias.npy")
    bias2 = np.load("/content/model_bias_3.npy")
    
    def extract_features(file_path):
        try:
            y, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=22)  # Extract 22 MFCC features
            return np.mean(mfcc.T, axis=0)  # Take mean over time
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    features = np.array([extract_features(file) for file in X])
    scaler = StandardScaler()
    features_scalaed = scaler.fit_transform(features)
    
    #model for distinguish balckboard
    black_logits = np.dot(features_scalaed, weights2.T) + bias2  
    black_pred = np.argmax(black_logits, axis=1) 
    
    #model for other objects
    original_logits = np.dot(features, weights1.T) + bias1  
    original_pred = np.argmax(original_logits, axis=1)   
    
    
    #integrate model for prediction
    final_pred = np.zeros_like(original_pred)
    for i in range(len(black_pred)):
        if black_pred[i] == 5:
            final_pred[i] = 5
        else:
            final_pred[i] = original_pred[i]

    return final_pred
    
    
    
    
    
    
    
    
    
    
    
    