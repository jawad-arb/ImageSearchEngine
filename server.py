import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__,static_url_path='/static')

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)

@app.route('/allProducts.html')
def all_products():
    return render_template('allProducts.html', img_paths=img_paths)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:10]  # Top 10 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        score_threshold = 1
        filtered_scores = [(score, path) for score, path in scores if score < score_threshold]
        return render_template('indexTest.html',
                               query_path=uploaded_img_path,
                               scores=filtered_scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run()