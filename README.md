# Links de archivos
## Vermont Data
```markdown
%%capture
!wget https://www.dropbox.com/scl/fi/uznju0fkgnwz1yi4cmgol/review-Vermont.json.gz?rlkey=4l3mithogu5c08x5lxe6pbb6t&st=wcprbsjo&dl=0

!wget https://www.dropbox.com/scl/fi/16vy3q077mz01n4r2nml1/meta-Vermont.json.gz?rlkey=zu5joaocmaqdg0p17oj14z5lz&st=kih98lc6&dl=0

!mv review-Vermont.json.gz?rlkey=4l3mithogu5c08x5lxe6pbb6t review-Vermont.json.gz

!mv meta-Vermont.json.gz?rlkey=zu5joaocmaqdg0p17oj14z5lz meta-Vermont.json.gz
```

## Vermont images 
```markdown
# Dataset con reviews de solamente restaurantes
!wget https://www.dropbox.com/scl/fi/cebo44ixv3003vba68pio/reviews_vermont.csv?rlkey=l272gentnbu9euyzixn32kig2&st=x461x04h&dl=0
````
```markdown
# Carpeta de imagenes formato png del dataframe
!wget https://www.dropbox.com/scl/fi/h5rjz0bmrqzfgpbr1d56t/new_images.zip?rlkey=00x7kxvw7qquloqvhxdai70dn&st=rv1qn7f3&dl=0

!mv new_images.zip?rlkey=00x7kxvw7qquloqvhxdai70dn images.zip
!mv reviews_vermont.csv?rlkey=l272gentnbu9euyzixn32kig2 reviews_vermont.csv
!unzip -qq images.zip
```

# Data 
## Images Embeddings
* Embeddings descargados en un archivo numpy para poder entrenar modelos CDL y VBPR (toma 11 minutos obtener las features en codigo). 
- `/data/imgs_features.npy`

