from flask import Flask, render_template, request
import os
import pandas as pd



"""
Se importa el dataset de peliculas a utilizar mediante el uso de la librearía Pandas
Al tratarser de un dataset en formato CSV, el método a utilizar es read_csv()
params: Path a Dataset
return: Dataset en formato DataFrame de Pandas

"""
peliculas = pd.read_csv('./imdb_top_1000.csv')

#print(display(peliculas))
#print(peliculas.shape)
#print(peliculas.columns)

"""
TRANSFORMACIONES NECESARIAS PARA EL ANÁLISIS:
1. Nos quedamos solo con las columnas que van a ser relevantes para el sistema de 
recomendación a implementar
2. Concatenar todo el reparte en una sola linea
3. Crear una nueva columna que contendrá todos los campos necesarios en formato string, 
concatenados por una ','
4. Nos quedamos solo con las columnas relevantes para el sistema de recomendación
a implementar
"""
peliculas = peliculas[['Series_Title','Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes']] #1

peliculas["Reparto"] = peliculas["Star1"] + ", " + peliculas["Star2"] + ", " + peliculas["Star3"] + ", " + peliculas["Star4"] #2
peliculas = peliculas[['Series_Title','Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Director', 'Reparto', 'No_of_Votes']]

peliculas['Etiquetas'] = peliculas['Runtime'] + ' ' + peliculas['Genre'] + ' ' + peliculas['Overview'] + ' ' + peliculas['Director'] + ' ' + peliculas['Reparto'] #3

peliculas = peliculas[['Series_Title', 'IMDB_Rating', 'Etiquetas', 'No_of_Votes']] #4

#print(display(peliculas))

""""
Importación de libreria SKlearn, quien nos provee todo el procesamiento de 
Machine learning de forma transparente, las operaciones y las distintas capas
utilizadas en el proceso de recomendación son invisibles a nosotros. 
"""
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(peliculas['Etiquetas']).toarray()

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vector)

#print(similarity)
#print(similarity.shape)

def recommend(movie):
    index = peliculas[peliculas['Series_Title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    a=list()
    for i in distances[1:6]:
        print("Titulo: " + str(peliculas.iloc[i[0]].Series_Title) + " Rating: " + str(peliculas.iloc[i[0]].IMDB_Rating))
        a.append("Titulo: " + str(peliculas.iloc[i[0]].Series_Title) + " Rating: " + str(peliculas.iloc[i[0]].IMDB_Rating))
    return a

#recommend('The Godfather')

app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(12)

@app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'POST':
        pelicula = request.form['pelicula']
        print(pelicula)
        recomendaciones = recommend(pelicula)
        print(recomendaciones)
        return render_template('index.html', recomendaciones=recomendaciones)
    return render_template('index.html')

@app.route("/who", methods=["GET", "POST"])
def who():
    return render_template('who.html')
if __name__ == '__main__':
    app.run(port=5540, debug=True)
    #init_gui(app, width=360, height=640, window_title="DaniCine" )
