# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:33:07 2020

@author: rodri
"""
import tensorflow as tf
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())

import keras
tf.compat.v1.keras.backend.set_session(sess) 

import flask
from flask import Flask, jsonify, request, render_template, redirect, g, flash, url_for, make_response, abort, Response, session
from jinja2 import Environment, FileSystemLoader
from urllib.parse import urljoin, urlparse, urldefrag, unquote
import json
import os
import requests
import cv2
import imageio as im
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from matplotlib import pyplot as plt
import seaborn as sns
from keras.models import model_from_json
from keras.models import load_model

sns.set(style='white', context='notebook', palette='deep')
print('\n libs carregadas \n')

global graph
keras.backend.clear_session()
#graph = tf.get_default_graph()
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = load_model('my_model_last')

print('\n modelo criado \n')

model.load_weights('model_weights_last.h5')
print('\n pesos carregados \n')

app = Flask(__name__)
UPLOAD_FOLDER = './static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print('\n servidor online carregados \n')

def make_prediction(imagem):
    img = cv2.resize(imagem, (100, 100))
    list_image = [normalize(img)]
    X = np.array(list_image)
    X = X.reshape(-1,100,100,1)
    predictions = model.predict_classes(X)
    print('predições')
    print(predictions)
    if predictions[0] == 0:
        return "Não apresenta Nódulo (96,37% de confiabilidade)"
    else:
        return "Atenção, Apresentou Nódulo! (82,66% de confiabilidade)"

def dicon_as_jpg(imagem, path):
    plt.imsave(path+'.png', imagem)
    return path+'.png'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    if request.files.get('file'):
        file = request.files['file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        print(path)
        imagem = im.imread(path).astype(np.float64)
        print('imagem lida')
        resultado = make_prediction(imagem)
        path2 = dicon_as_jpg(imagem, path)
        return render_template('predictions.html', imagem=path2, resultado = resultado)
    elif request.form.get('paste'):
        return Response( f"<h2> Execute </h2>")

    
if __name__ == '__main__':
    app.run()
    

#http://127.0.0.1:5000/predict
#path = r'C:\Users\rodri\2 Mestrado\Programa - medicina\New Test\static\000070.dcm'
#path = r'~/Downloads/Mestrado/imagens/nodulo/000106.dcm'