from keras.applications import mobilenet_v2
import numpy as np
import imageio
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import decode_predictions
from operator import itemgetter

sep_root_path = './models'
model = moblienet = mobilenet_v2.MobileNetV2(weights = 'imagenet')

def classify(filename):
    test_dir = 'uploaded'
    data = np.empty((1, 224, 224, 3))
    img = imageio.imread(os.path.join(test_dir, 'image', filename))
    data[0] = np.resize(img, (224, 224, 3))
    result_list = []
    predictions = model.predict(data)
    for name, desc, score in decode_predictions(predictions)[0]:
        result_list.append([desc, 100 * score])
    sorted_list = sorted(result_list, key=itemgetter(1), reverse=True)
    
    return sorted_list[0]

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False
app.config['UPLOAD_FOLDER'] = 'uploaded\image'

@app.route('/API', methods=['POST', 'GET'])
def pred():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        val1 = classify(f.filename)
        return jsonify(val1)
    if request.method == 'GET':
        return "get!"
    
if __name__ == '__main__':
    app.run()