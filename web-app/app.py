from flask import Flask, request, Response
import numpy as np
import cv2
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from model import model
app = Flask(__name__)
model_obj = model()
@app.route('/api/predict', methods=['POST'])
def test():
    r = request
    nparr = np.fromstring(r.data,dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    pred = model_obj.predict(img)
    return pred
if __name__ == '__main__':
    app.run()
