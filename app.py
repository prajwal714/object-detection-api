
import io

from flask import Flask, render_template, request, jsonify

from PIL import Image

import cv2 as cv
import numpy

global cvNet
cvNet = cv.dnn.readNetFromTensorflow('./model/frozen_inference_graph.pb',
                                             './model/ssd_mobilenet_v1_coco_2017_11_17.pbtxt')

classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

app = Flask(__name__)


       
@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def upload():
    data = {"success":False}
    if request.method == 'POST':
        if request.files.get("image"):
            image=request.files["image"].read()
            image = Image.open(io.BytesIO(image))
           
            data["predictions"]=[]
            width=300
            height=300
            img = cv.cvtColor(numpy.array(image), cv.COLOR_BGR2RGB)
            img = cv.resize(img,(width,height))
           
           
            print ("Width ,height :",width,height)
            cvNet.setInput(cv.dnn.blobFromImage(img, 0.007843,(width,height),(127.5, 127.5, 127.5), swapRB=True, crop=False))
            detections = cvNet.forward()
            cols = img.shape[1]
            rows = img.shape[0]
    
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    class_id = int(detections[0, 0, i, 1])
    
                    xLeftBottom = int(detections[0, 0, i, 3] * cols)
                    yLeftBottom = int(detections[0, 0, i, 4] * rows)
                    xRightTop = int(detections[0, 0, i, 5] * cols)
                    yRightTop = int(detections[0, 0, i, 6] * rows)
                    print (xLeftBottom,yLeftBottom)
                    print (xRightTop,yRightTop)
                    x = (xLeftBottom+xRightTop)/2
                    y = (yLeftBottom+yRightTop)/2
                   
                    print(classNames[class_id])
                    print("centroid :",x,y)
                    xfinal=x
                    yfinal=y
                    if xfinal <= width/3:
                            W_pos = "left "
                    elif  xfinal <= (2*width/3):
                            W_pos = "center "
                    else:
                           W_pos = "right "
                        
                    if yfinal <= (height/3):
                            H_pos = "bottom "
                    elif yfinal <= (2*height/3):
                            H_pos = "mid "
                    else:
                            H_pos = "top"
                    
                    cv.rectangle(img, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                                 (0, 0, 255))
                    if class_id in classNames:
                        
                        r={"label": classNames[class_id],"probability":str(confidence),"x":W_pos,"y":H_pos}
                        data["predictions"].append(r)
     #                   labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
      #                  yLeftBottom = max(yLeftBottom, labelSize[1])
       #                 cv.putText(img, label, (xLeftBottom+5, yLeftBottom), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
                        
        #img = cv.imencode('.jpg', img)[1].tobytes()
            data["success"]=True
       # file = Image.open(request.files['file'].stream)
           
    #print(data)        
    return jsonify(data)
     #send_file(io.BytesIO(img),attachment_filename='image.jpg',mimetype='image/jpg')


if __name__ == "__main__":
   
    
    app.run()
