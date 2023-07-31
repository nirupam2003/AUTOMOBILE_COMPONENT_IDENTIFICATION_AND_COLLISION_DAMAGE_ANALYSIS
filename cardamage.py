from flask import request,Flask,jsonify
from ultralytics import YOLO
from itertools import compress
import numpy as np
from PIL import Image
import base64
import io
import cv2

app=Flask(__name__)
'''def store(result,model,result1,model1):
        damage=model.names
        parts=model1.names
        part_ind=result1[0].boxes.cls
        damage_ind=result[0].boxes.cls'''
def calculate_iou(box1, box2):
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    #If there's no intersection, return 0
    if x_min_inter >= x_max_inter or y_min_inter >= y_max_inter:
         return 0.0

    # Calculate the area of the intersection rectangle
    intersection_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    print(box1_area)
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    print(box2_area)
    # Calculate the area of the union of the two boxes
    union_area = box1_area + box2_area - intersection_area
    print("u:",union_area)
    # Calculate the Intersection over Union
    iou = intersection_area / union_area

    return iou


            
@app.route('/predict',methods=['POST'])
def predict():
    if 'img' not in request.files:
            return jsonify({'error': 'No files uploaded or invalid key. Please use the key "file" for uploading.'})
    data = request.files['img'].read()
    if not data:
            return jsonify({'error': 'No files uploaded'})
    else:
        
        print(data)
        fimg=io.BytesIO(data)
        a = Image.open(fimg)
        img = np.array(a)
        print('a', np.array(a))
        
        img_rgb=a.convert("RGB")
        model=YOLO('C:\\Users\\HP\\Downloads\\type of damage.pt')
        result=model(img_rgb,save=True)
        damage=model.names
        del model
        
        model1=YOLO("C:\\Users\\HP\\Downloads\\parts.pt")
        result1=model1(img_rgb,save=True,classes=[0,1,2,3,4,5,6,7,9])
        parts=model1.names
        del model1
        c=result1[0].boxes.xyxy
        d=result1[0].boxes.cls
        cord=list(compress(c,d))
        cord_part=[[int(x) for x in list] for list in cord]#cord_part and d has the corresponding cordinates and their parts
        if cord_part==[]:
            return "no parts detected by model"
        else:
            pass
        a=result[0].boxes.xyxy
        b=result[0].boxes.cls
        print(b)
        cord_brok=[[int(x) for x in list] for list in a]    
        if cord_brok==[]:
            return "damage not detected by model"
        else:
            pass
        if(cord_brok==[]):
             return "MODEL COULD NOT IDENTIFY DAMAGE"
        else:
                out=[]
                for i in range(len(cord_brok)):
                        for j in range(len(cord_part)):
                              p1, q1, p2, q2 = cord_brok[i]
                              x1, y1, x2, y2 = cord_part[j]
                              box1 = (p1, q1, p2, q2)  # (x_min, y_min, x_max, y_max)
                              box2 = (x1, y1, x2, y2)  # (x_min, y_min, x_max, y_max)
                              iou_value = calculate_iou(box1, box2)
                              # print()
                              if(iou_value>0.0):
                                print(damage[int(b[i])],'near the',parts[int(d[j])])
                                out.append({'part':parts[int(d[j])],
                                            'damage':damage[int(b[i])],
                                            'iou':iou_value})
                                print(out)
                                print("IoU:", iou_value)
                                if (out==[]):
                                     return "NO DAMAGE DETECTED"
                                else:
                                     return jsonify(out)
        
        
    

if (__name__=="__main__"):
    app.run(debug=False, port=7000)