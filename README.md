# Sistemas de recomendación de restaurantes basados en modelos multimodales
## Descripcion general
El siguiente proyecto contiene todos los codigos relacionados a al proyecto de recomendacion
de restaurantes utilizando modelos multimodales. En este caso, utilizamos LLMs como LLaVA y BLIP.

## Autores
| Nombre         | GitHub          | Correo                  |
|----------------|-----------------|-------------------------|
| Felipe Torres  | @PipeXtz        | felipetorresp@uc.cl |
| Kahil Rasse    | @K-hil          | kahil.rasse@uc.cl   |
| Pedro Zavala   | @pedrozavalat   | pedropablozavalat@uc.cl  |

## Tabla de contenidos
1. [Carpetas y archivos](#carpetas-y-archivos)
2. [Consideraciones generales](#consideraciones-generales)
    
    2.1. [Baselines models](#random-most-popular)
    
    2.2. [CDL y VBPR](#cdl-y-vbpr)
    
    2.3. [Tiny LLaVA - KNN](#llava)
    
    2.4. [BLIP - KNN](#blip)
    
    2.5. [LightFM](#lightfm)

3. [Ejecuciones de archivos](#ejecuciones-de-archivos)
4. [Dropbox](#links-dropbox)


# Carpetas y archivos
> 📌  Guiarse con la tabla de contenidos de cada google colab.
* `iic3633_BLIP.ipynb` y `iic3633_LLaVA_Tiny 2.ipynb`: contienen la definicion de los modelos, analisis de parametros, resultados basados en metricas de evaluacion y ejemplos de las listas de recomendacion. 

* `iic3633_lightFM.ipynb`: contiene la experimentacion y generacion de listas de recomendacion utilizando lightFM con las descripciones realizadas por LLaVA y lightFM utilizando los comentarios de las reseñas. 

* `iic3733_v1_models_baselines.ipynb`: contiene la experimentacion y generacion de listas de recomendacion utilizando modelos baselines, como Random y MostPopular. Además contiene otros modelos, como VBPR y CDL. 

* `iic3633_review_generation_v2.ipynb`. contiene las explicaciones y generaciones de recomendaciones. 


* `/utils`: carpeta que contiene codigo necesarios para la formacion del conjunto de datos.
    * `iic3633_data_download.ipynb`: codigo que descarga informacion de las reviews de un estado y  guarda las imagenes relacionadas y un dataframe con el estado solicitado. 

    * `iic3633_merge_reviews.ipynb`: codigo que permite realizar un *merge* de los conjuntos de datos de los dataframes de todas los estados. 
    
    * `iic3633_restaurants_datagen.ipynb`: codigo que permitio la generacion de un dataframe completo que une las reviews de todas los estados. En este sentido, concatena las filas de las reviews de los estados en una sola tabla. 
    
* `/plots`: carpeta con graficos de comparacion entre los resultados de los modelos.

# Consideraciones generales
Antes de obtener las listas de recomendaciones se tienen que tomar las siguientes consideraciones. 
*  El conjunto de datos de las reviews, metadata, restaurantes, descripciones de imagenes con VG19, LLaVA y BLIP ya
estan descargadas y se encuentran almacenados en un dropbox para llegar e importar. No es necesario importar nada: [ver links](#links-dropbox)

# Ejecuciones de archivos
### Random, Most Popular
1. Ejecutar codigos Set Up Libraries, Data Preprocessing y Utils seguidamente. 

2. Para ver los resultados de Most Popular y Random, solo se requiere ejecutar todas sus celdas respectivas (de su seccion).  


### CDL y VBPR
1. Ejecutar codigos Set Up Libraries, Data Preprocessing y Utils seguidamente. 
2. Para ver los resultados de VBPR y CDL se debe ejecutar todas las celdas respectivas. **Ojo**: para importar directamente los embeddings de las imagenes -y no extraerlos, ya que toma bastante tiempo- y usarlos en el modelo VBPR, se 
debe definir la variable `D_IE = 0`. 
3. Las metricas de evaluacion de CDL toman alrededor de 10 minutos. 

### LLaVA 
1. Ejecutar codigos Set Up Libraries, Data Preprocessing y Utils seguidamente. 
2. En la seccion **Models**, se debe ejecutar primero el codigo 
    ```python
    # all-MiniLM-L6-v2
    '''Sentence transformers AllMiniLM para codificar las descripciones hecha por 
    LLaVA'''
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    ```
    Luego, ejecutar todo el codigo correspondiente a la seccion **Download Imgs Descriptions** (para asi evitar la extraccion de descripciones de las imagenes, ya qur toma 2h). 
3. Ejecutar funciones de **Training functions**. 
4. En **Sensitivity Analysis**, ejecutar
    ```python
    k_values = [5, 10, 20, 30, 50]
    llava_most_sim_reclist = llava_train(train_users, test_users, mode='knn', k_values=k_values)
    ```
    ```python
    llava_mean_reclist = llava_train(train_users, test_users, mode='mean', k_values=k_values)
    ```
5. Para ver resultados y ejemplos, ejecutar codigos de las subsecciones **Metrics** y **Examples** de **Results**

### BLIP

1. Ejecutar codigos Set Up Libraries, Data Preprocessing y Utils seguidamente. 
2. Ejecutar **BLIP/Description's model download**, para descargar las descripciones de imagenes realizadas por BLIP. 
3. En **Sensitivity Analysis**, ejecutar:
    ```python
    k_values = [5, 10, 20, 30, 50]
    blip_most_sim_reclist = blip_train(train_users, test_users, mode='knn', k_values=k_values)
    blip_mean_reclist = blip_train(train_users, test_users, mode='mean', k_values=k_values)
    ```
4. Para ver resultados y ejemplos, ejecutar codigos de las subsecciones **Metrics** y **Examples** de **Results**

### LightFM
1. Ejecutar codigos Set Up, Libraries, Data Preprocessing y Utils seguidamente. 
2. En **Models/Multimodal Recsys**, ejecutar solo el codigo
    ```python 
    # all-MiniLM-L6-v2
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    ```
3. Ejecutar todo el codigo de **Download Imgs Descriptions**
4. Para obtener resultados y metricas del modelo *LightFM + LLaVA*, se debe ejecutar las secciones de **Evaluation and Examples** directamente.
    Si se quiere obtener *LightFM Basic* (usando texto de las reseñas), se debe ejecutar el codigo que se encuentra al inicio de la seccion **Model LightFM + allMiniLLM** y antes de **Sensitivity Analysis**. Despues, se ejecuta el codigo de **Evaluation and Examples**. 



# Links Dropbox
```markdown
%%capture
# Reviews images related
!wget https://www.dropbox.com/scl/fi/2o3fmzj4jdsfujfhu0mns/reviews.zip?rlkey=yc8doasvaavp2993huknxw6jt&st=3e3bpy5q&dl=0
!mv reviews.zip?rlkey=yc8doasvaavp2993huknxw6jt reviews.zip
!unzip reviews.zip
```
```markdown
# Numpy array de embeddings de imagenes (CDL y VBPR)
!wget https://www.dropbox.com/scl/fi/rd8xxbz8duqp7s3nkzylv/imgs_features.npy?rlkey=19e6k5orsnks9kn9rwhw763f2&st=9mn8cev0&dl=0
!mv imgs_features.npy?rlkey=19e6k5orsnks9kn9rwhw763f2 imgs_features.npy
```
```markdown
# Metadata for each restaurant
!wget https://www.dropbox.com/scl/fi/cxckzuj81gsnlsvclqnza/metadata.json.gz?rlkey=d4xerrcwbeyt09oi01f9f4wru&st=sv6cnpzh&dl=0
!mv metadata.json.gz?rlkey=d4xerrcwbeyt09oi01f9f4wru metadata.json.gz
```
```markdown
# CSV de restaurantes
!wget https://www.dropbox.com/scl/fi/g8862obe2z29su61popjx/restaurants.csv?rlkey=rhrn6vg0zg6ier2yuz9lh00yh&st=fqh0067i&dl=0
!mv restaurants.csv?rlkey=rhrn6vg0zg6ier2yuz9lh00yh restaurants.csv
```
```markdown
# CSV de reviews 
!wget https://www.dropbox.com/scl/fi/6u1yfcnnf4jqmhedx519u/reviews.csv?rlkey=xqmvvohkq0i0k7hho79fs43b6&st=ko3q9dnq&dl=0
!mv reviews.csv?rlkey=xqmvvohkq0i0k7hho79fs43b6 reviews.csv
```
```markdown
# TINY LLaVA Descriptions json.gz
!wget https://www.dropbox.com/scl/fi/50pmwvytozpz0cl1p054f/tiny_LLaVa_images_descriptions.json.gz?rlkey=7vreygmtd16lohs3bx6yvmwdk&st=9568qz84&dl=0
!mv tiny_LLaVa_images_descriptions.json.gz?rlkey=7vreygmtd16lohs3bx6yvmwdk tiny_LLaVa_images_descriptions.json.gz
```
```markdown
# BLIP LLaVA Descriptions json.gz
!wget https://www.dropbox.com/scl/fi/j3m7xv2811l6v09v646ty/BLIP_images_descriptions.json.gz?rlkey=d3an8z48kpp1eyzj8v5k197s1&st=c7t76xpf&dl=0
!mv BLIP_images_descriptions.json.gz?rlkey=d3an8z48kpp1eyzj8v5k197s1 BLIP_images_descriptions.json.gz
```
