
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio as im
import os
import time
import seaborn as sns
import cv2
sns.set(style='white', context='notebook', palette='deep')


from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# import Sequential from the keras models module
from keras.models import Sequential

# import Dense, Dropout, Flatten, Conv2D, MaxPooling2D from the keras layers module
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, roc_curve, classification_report, recall_score, precision_score, f1_score
import itertools


from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


cases = pd.read_csv(r'lista.csv')
cases.head(10)


loaded_images = pd.DataFrame(columns=['scan', 'slice'])
list_images = []
loaded_images

def leitura_update(path):
    
    url_base = path

    global loaded_images
    global list_images

    start = time.time()

    for file_name in os.listdir(url_base):
        
        start = time.time()
        
        for file2 in os.listdir(url_base+'/'+file_name):
            for file3 in os.listdir(url_base+'/'+file_name+'/'+file2):
                for file4 in os.listdir(url_base+'/'+file_name+'/'+file2+'/'+file3):

                    if file4.endswith('.dcm'):
                        img = im.imread(os.path.join(url_base, file_name, file2, file3, file4)).astype(np.float64)
                        img_dic = img.meta
                        scan = img_dic['SeriesNumber']
                        slice_n = img_dic['InstanceNumber']
                        loaded_images.loc[len(loaded_images)] = [scan, slice_n]
                        img = cv2.resize(img, (100, 100)) # redução dimensional
                        list_images.append(img)
    
        end = time.time()-start

        print('Levou', round(end,2), 'segundos para executar na pasta', file_name)

url_base1 = r'pasta_de_fotos'
leitura_update(url_base1)

# Salvando as fotos carregadas
import pickle

with open('list_images_2020', 'wb') as f:
    pickle.dump(list_images, f)

with open('loaded_images_2020', 'wb') as f:
    pickle.dump(loaded_images, f)


# para carregar
import pickle

with open(r'list_images_2020', 'rb') as f:
    list_images = pickle.load(f)

with open(r'loaded_images_2020', 'rb') as f:
    loaded_images = pickle.load(f)

print(len(list_images))
print(len(loaded_images))

normalized_list = []
for i in list_images:
    normalized_list.append(normalize(i))

X = np.array(normalized_list)
X.shape
X = X.reshape(-1,100,100,1)
X.shape


loaded_images.head()

to_add = cases[['scan', 'slice']].copy()
to_add['has_tumor'] = 1
to_add.head()


to_add = to_add.drop_duplicates(subset=['scan', 'slice'], keep='last')


loaded_images_grouped = pd.merge(loaded_images,
                 to_add,
                 on=['scan','slice'],
                 how = 'left',
                                validate = 'm:1')
loaded_images_grouped.fillna(0, inplace=True)
loaded_images_grouped.head()
to_add.head()

loaded_images_grouped.info()
loaded_images_grouped['has_tumor'].value_counts().plot(kind='pie');
loaded_images_grouped['has_tumor'].value_counts()



loaded_images_grouped['ID'] = np.arange(len(loaded_images_grouped))


tumor = loaded_images_grouped[loaded_images_grouped.has_tumor==1]
not_tumor = loaded_images_grouped[loaded_images_grouped.has_tumor==0]


from sklearn.utils import resample
tumor_upsampled = resample(tumor,
                          replace=True, # sample with replacement
                          n_samples=750, # match number in majority class
                          random_state=12) # reproducible results

not_tumor_downsampled = resample(not_tumor,
                          replace=False, # sample with replacement
                          n_samples=750, # match number in majority class
                          random_state=12) # reproducible results


resampled = pd.concat([tumor_upsampled, not_tumor_downsampled])


X_resampled = []
for row in resampled.ID:
    X_resampled.append(X[row])
X_resampled = np.array(X_resampled)
X_resampled.shape


y = resampled['has_tumor'].values
y = np.array(y)
y = to_categorical(y, num_classes = 2)    # tem tumor = [0,1], sem tumor = [1,0]


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y, test_size = 0.3, random_state=12, stratify=y)

print(X_train.shape, 'train samples')
print(X_test.shape, 'test samples')
print(y_train.shape, 'respostas do treino')
print(y_test.shape, 'respostas do teste')


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (2,2),padding = 'Same', 
                 activation ='relu', input_shape = (100,100,1)))

model.add(Flatten())
model.add(Dense(10, activation = "relu"))
model.add(Dense(2, activation = "softmax"))   #NUMERO DE OUTPUTS

model.summary()


# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# Compile the model
model.compile(optimizer = optimizer
              , loss = "categorical_crossentropy"
              , metrics=["accuracy"])


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=0, 
                                            factor=0.5, 
                                            min_lr=0.00001)


epochs = 45
batch_size = 40

# data augmentation to prevent overfitting

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=3,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.05, # Randomly zoom image 
        width_shift_range=False,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=False,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)


# Fit the model
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, 
                              validation_data = (X_test,y_test),
                              verbose = 0, 
                              steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1, figsize=(9, 5))
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


predictions = model.predict_classes(X_test)


print('Acuracidade do modelo: ',round(accuracy_score(predictions,np.argmax(y_test, axis = 1))*100,2), '%')
print('Demais Indicadores: \n')
print('   recall_score: ',round(recall_score(predictions,np.argmax(y_test, axis = 1)),2))
print('   precision_score: ',round(precision_score(predictions,np.argmax(y_test, axis = 1)),2))
print('   f1_score: ',round(f1_score(predictions,np.argmax(y_test, axis = 1)),2))
print('   roc: ',round(roc_auc_score(predictions,np.argmax(y_test, axis = 1)),2))


print(classification_report(predictions,np.argmax(y_test, axis = 1)))


def plot_confusion_matrix(cm, target_names, title='Confusion matrix'):
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    cmap = plt.get_cmap('Blues')
    plt.rcParams.update(plt.rcParamsDefault)
    %matplotlib inline
    
    #plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.3f}; misclass={:0.3f}'.format(accuracy, misclass))
    plt.show()


plot_confusion_matrix(confusion_matrix(predictions,np.argmax(y_test, axis = 1)),target_names=['tumor','not_tumor'])

# predict probabilities
probs = model.predict(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(np.argmax(y_test, axis = 1), probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis = 1), probs)
# plot
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# Salvando os pesos do modelo com bom desempenho

model.save_weights('model_weights_last.h5')


# Salvando o modelo em si

model.save('my_model_last')



# save as JSON
import json
json_string = model.to_json()

with open('model.json','w') as f:
    json.dump(json_string, f)



    # para carregar

'''
model.load_weights('my_model_weights.h5')
from keras.models import load_model
new_model = load_model('my_model')
''';


# Display some error results 

# Errors are difference between predicted labels and true labels
errors = (predictions - np.argmax(y_test, axis=1) != 0)
print("De um total de",len(predictions), "errou", sum(errors), "vezes")


# Dados de entrada dos previstos errados
X_test[errors].shape

# Ajustando dimensões para uma imagem
X_test[errors][37].reshape(100,100).shape

# Acessando os originais
np.argmax(y_test[errors], axis=1).shape

# Acessando a resposta original de alguma previsão
np.argmax(y_test[errors], axis=1)[1]


final_table = pd.DataFrame({'previsao': predictions, 'real':np.argmax(y_test, axis=1)})
print('falso positivo')
final_table[final_table['previsao'] > final_table['real']]

print('falso negatvo')

final_table[final_table['previsao'] < final_table['real']]


import matplotlib.image as mpimg
import seaborn as sns
import cv2
sns.set(style='white', context='notebook', palette='deep')

print('Imagens que o sistema previu errado, com sua resposta original \n')
fig, ax = plt.subplots(2,3, figsize=(16, 10), sharex=True,sharey=True)
ax[0,0].imshow(X_test[14].reshape(100,100))
ax[0,0].set_title("Exemplo de Falso Positivo")

ax[0,1].imshow(X_test[449].reshape(100,100))
ax[0,1].set_title("Exemplo de Falso Positivo")

ax[0,2].imshow(X_test[434].reshape(100,100))
ax[0,2].set_title("Exemplo de Falso Positivo")

ax[1,1].imshow(X_test[51].reshape(100,100))
ax[1,1].set_title("Exemplo de Falso Negativo")      

ax[1,0].imshow(X_test[321].reshape(100,100))
ax[1,0].set_title("Exemplo de Falso Negativo")

ax[1,2].imshow(X_test[417].reshape(100,100))
ax[1,2].set_title("Exemplo de Falso Negativo")

ax[0,0].axis('off')
ax[1,0].axis('off')
ax[1,2].axis('off')
ax[0,1].axis('off')
ax[1,1].axis('off')
ax[0,2].axis('off')


plt.show()


# Display some correct results 

acertos = (predictions - np.argmax(y_test, axis=1) == 0)
print("De um total de",len(predictions), "acertou", sum(acertos), "vezes")


final_table = pd.DataFrame({'previsao': predictions, 'real':np.argmax(y_test, axis=1)})
print('trues')
final_table[final_table['previsao'] == final_table['real']]


import matplotlib.image as mpimg
import seaborn as sns
import cv2
sns.set(style='white', context='notebook', palette='deep')

print('Imagens que o sistema previu errado, com sua resposta original \n')
fig, ax = plt.subplots(2,3, figsize=(16, 10), sharex=True,sharey=True)
ax[0,0].imshow(X_test[0].reshape(100,100))
ax[0,0].set_title("Exemplo de Verdadeiro Positivo")

ax[0,1].imshow(X_test[2].reshape(100,100))
ax[0,1].set_title("Exemplo de Verdadeiro Positivo")

ax[0,2].imshow(X_test[4].reshape(100,100))
ax[0,2].set_title("Exemplo de Verdadeiro Positivo")

ax[1,1].imshow(X_test[17].reshape(100,100))
ax[1,1].set_title("Exemplo de Verdadeiro Negativo")      

ax[1,0].imshow(X_test[15].reshape(100,100))
ax[1,0].set_title("Exemplo de Verdadeiro Negativo")

ax[1,2].imshow(X_test[6].reshape(100,100))
ax[1,2].set_title("Exemplo de Verdadeiro Negativo")

ax[0,0].axis('off')
ax[1,0].axis('off')
ax[1,2].axis('off')
ax[0,1].axis('off')
ax[1,1].axis('off')
ax[0,2].axis('off')


plt.show()








