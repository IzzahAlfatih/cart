from flask import Flask, jsonify, request
from ultralytics import YOLO
#import firebase_admin
import json
#from firebase_admin import storage, credentials
import pandas as pd
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

# # Initialize Firebase Admin SDK with credentials
# cred = credentials.Certificate("tubescc2023-firebase-adminsdk-a4hvb-b62ff9d985.json")#path to key
# firebase_admin.initialize_app(cred, {"storageBucket": "tubescc2023.appspot.com"})# initialize storage bucket

# def upload_to_firebase():
# # Initialize Firebase Storage client
#      bucket = storage.bucket()
#      result=('testing')#result for send as json this time for test purpose using string
#      result_blob = bucket.blob("results/result.json")
#      result_blob.upload_from_string(json.dumps(result), content_type="application/json")
#      return print("json has been uploaded to firebase")

def predict(img_path):

    # Load the YOLOv8 model
    model = YOLO('10productnano.pt')

    # Run inference
    results = model(img_path)

    # translate boxes data from a Tensor to the List of boxes info lists
    boxes_list = results[0].boxes.data.tolist()
    columns = ['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class_id']

    # iterate through the list of boxes info and make some formatting
    for i in boxes_list:
      # round float xyxy coordinates:
        i[:4] = [round(i, 1) for i in i[:4]]
      # translate float class_id to an integer
        i[5] = int(i[5])
    # add a class name as a last element
        i.append(results[0].names[i[5]])

    # create the result dataframe
    columns.append('class_name')
    result_df = pd.DataFrame(boxes_list, columns=columns)

    # sum of every class detected
    product_list = ['89686723021', '12', '13', '14']
    
    sum_product = result_df.class_name.value_counts().to_dict()
    for i in product_list:
        if i not in sum_product:
            sum_product[i] = 0
    
    return sum_product

def base64_to_image(base64input):
    img = Image.open(BytesIO(base64.decodebytes(bytes(base64input, "utf-8"))))
    return img

@app.route('/predict', methods = ['POST'])
def predictionHandler():
    if request.method == "POST":
      if request.json is None:
        return jsonify({"error" : "no JSON"})
      try:
          content = request.get_json()
          image = base64_to_image(content['input_image'])
          pred_result = predict(image)

          return pred_result#jsonify({"code": 200,"jumlah_barang":pred_result,"message": "JSON Posted"})
        
      except Exception as e:
          return jsonify({"code": 400, "message": str(e)})

    return jsonify({"error": False, "message": "Prediction service online. Try POST method to predict an image file."})

if __name__ == '__main__':
    app.run()
