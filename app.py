import numpy as np
import faiss
import pickle
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

app = Flask(__name__)

faiss_index = None
labels_for_index = None
names_for_index = None

try:
    faiss_index = faiss.read_index('Dataset/lfw_embeddings_index.faiss')
    
    if isinstance(faiss_index, faiss.Index):
        print("FAISS index loaded successfully.")
    else:
        raise TypeError("The loaded FAISS index is not of type 'faiss.Index'.")

    # Load labels and names
    with open('Dataset/lfw_labels_for_index.pkl', 'rb') as f:
        labels_for_index = pickle.load(f)

    with open('Dataset/lfw_names_for_index.pkl', 'rb') as f:
        names_for_index = pickle.load(f)

except Exception as e:
    print(f"Error loading FAISS index or metadata: {e}")
    faiss_index = None

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def preprocess_image(img):
    """
    Preprocesses the input image, converting grayscale to RGB, resizing, and applying preprocess_input
    """
    if img.mode != 'RGB':
        img = img.convert('RGB') 
    
    img = img.resize((224, 224)) 
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if not faiss_index:
        return "Error: FAISS index not loaded properly."

    file = request.files['file']
    if file:
        try:
            img = Image.open(BytesIO(file.read()))
            img_array = preprocess_image(img)

            embedding = model.predict(img_array)
            embedding = embedding.flatten()

            D, I = faiss_index.search(np.array([embedding]), k=5)

            results = [(D[0][i], names_for_index[I[0][i]]) for i in range(len(I[0]))]

            results.sort(key=lambda x: x[0])

            results_for_redirect = '|'.join([f"{name}:{distance}" for distance, name in results])

            return redirect(url_for('results_page', results=results_for_redirect))

        except Exception as e:
            return f"Error processing image: {e}"

    return "No file uploaded."

@app.route('/results')
def results_page():
    results_string = request.args.get('results')

    if results_string:
        results = [(float(item.split(':')[1]), item.split(':')[0]) for item in results_string.split('|')]
    else:
        results = []

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
