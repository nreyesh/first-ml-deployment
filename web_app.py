import cv2
import numpy as np
from PIL import Image
import requests
import streamlit as st
from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.image import img_to_array

### Funciones Auxiales para manejro de imagenes
# Preprocesamiento de la imagen 
def load_img_file(path):
    size = (96,96)
    image = Image.open(path)
    image = np.array(image)
    image = cv2.resize(image,size,interpolation=cv2.INTER_CUBIC)
    image = image.astype("float") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Descarga imagen de una url
def load_img_url(url):
  response = requests.get(url, stream=True).raw
  image = load_img_file(response)
  return image

# Obtener prediccion
def prediction(img, model):
    pred = model.predict(img)
    pred_idx = int(np.argmax(pred, axis=1))
    pred_prob = pred[0][pred_idx]
    return pred_idx, pred_prob

############### -------------------------- ###############   
### Inicio Aplicacion
# Carga del Modelo
def model_loader():
  model = load_model('final-model.hdf5')
  return model

# Ver imagen
def ver(image,model,categorias):
  st.image(image, width=200, channels="RGB")
  result = st.button('Clasificar')
  if result:
    pred_cat,pred_prob = prediction(image, model)

    st.write('La imagen corresponde a:', categorias[pred_cat])
    st.write('La probabildiad asociada a esta predicción es:',pred_prob)
    print('La imagen corresponde a',categorias[pred_cat], 'con una probabilidad de',pred_prob)

# Intefaz grafica para interactuar con la persona
st.title('Clasificador Imágenes de Ropa')
with st.spinner('Modelo está siendo cargado..'):
  model = model_loader()
 
# Cargar Imagen
tipos = ['URL', 'LOCAL']
select_box = st.selectbox('¿Dónde está ubicada la imagen que deseas clasificar?', options=tipos)
categorias = ['VESTIDO','JEANS','POLERA']

# mediante path
if select_box == 'LOCAL':
  file = st.file_uploader('Carga tu imagen para que sea clasificada dentro de tres posibles categorías: jeans, polera o  vestido. ', type=['jpg', 'png','jpeg'])
  
  if file is None:
    st.text('Por favor cargue una imagen')
  else:
    image = load_img_file(file)
    ver(image,model,categorias)

else:  # mediante url
  url = st.text_input('Ingrese la URL de la imágen a clasificar','https://img.ltwebstatic.com/images3_pi/2022/05/04/16516458824f79b0f4e627581293a1eaa84af7a0a9_thumbnail_600x.webp')
  
  if url is None:
    st.text('Por favor cargue una imagen')
  else:
    image = load_img_url(url)
    ver(image,model,categorias)
