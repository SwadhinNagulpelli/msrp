from flask import Flask, render_template, request
import tensorflow as tf 
import tensorflow as tf 
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
import numpy as np

saved_model = tf.keras.models.load_model('model\\saved_model_nn2')

app = Flask(__name__,template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text1 = request.form['text1']
        text2 = request.form['text2']
        similarity = final_test(text1, text2, saved_model)
        return render_template('result.html', text1=text1, text2=text2,similarity = similarity)
    return render_template('form.html')

def final_test(t1,t2,model): 
  import re
  lemm = WordNetLemmatizer()
  corpus = []
  concat = "-".join([t1, t2])
  review = re.sub("[^a-zA-Z0-9]"," ",concat).lower().split()
  review = [lemm.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
  corpus.append(" ".join(review))
  cv = CountVectorizer(max_features=2500)
  X = cv.fit_transform(corpus).toarray()
  X.resize((1,2500),refcheck=False)
  y_predicted = model.predict(X)
  y_predicted = y_predicted.flatten()
  #y_predicted = np.where(y_predicted > 0.5, 1, 0)
  return y_predicted[0]



if __name__ == '__main__':
    app.run(debug=True)