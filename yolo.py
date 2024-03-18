import numpy as np
import time
import cv2

path_of_image = "images/india.jpg"
def_confidence_score = 0.5
threshold = 0.3

box_array = []
conf_array = []
ID_array = []

coco_dataset_ = "yolo-coco/coco.names"

with open(coco_dataset_, 'r') as file:
    LABELS = [line.strip() for line in file]

COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

image = cv2.imread(path_of_image)
cv2.imshow("Previous Image", image)
cv2.waitKey(0)

if image is None:
    print("[ERROR] Wrong Image path or image not available")
    exit()

img_height = image.shape[0]
img_weight = image.shape[1]

weightsPath = 'yolo-coco/yolov3.weights'
configPath = 'yolo-coco/yolov3.cfg'
# u_layer means unconnected layer
drknet = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
u_layer = drknet.getUnconnectedOutLayers()
u_layer = u_layer.tolist()
l_name = drknet.getLayerNames()
ln = [l_name[i - 1] for i in u_layer]

binary_large_object = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416))
drknet.setInput(binary_large_object)
final_output = drknet.forward(ln)

for output in final_output:
    for suppression in output:
        if len(suppression) < 6:
            print("Skipping this due to Invalid set of data")
            continue

        scores = suppression[5:]
        classID = np.argmax(scores)
        if classID >= len(scores) or classID < 0:
            print("Skipping this due to Invalid Class ID")
            continue

        confidence = scores[classID]
        if confidence > def_confidence_score:
            box = suppression[0:4] * np.array([img_weight, img_height, img_weight, img_height])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            box_array.append([x, y, int(width), int(height)])
            conf_array.append(float(confidence))
            ID_array.append(classID)

non_max_suppression = cv2.dnn.NMSBoxes(box_array, conf_array, def_confidence_score, threshold)

if len(non_max_suppression) > 0:
    for i in non_max_suppression.flatten():
        x, y, w, h = box_array[i]
        classID = ID_array[i]
        confidence = conf_array[i]
        clr = [int(c) for c in COLORS[classID]]
        confidence_box_color = (255, 255, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), clr, 2)
        cv2.rectangle(image, (x, y - 20), (x + 100, y), confidence_box_color, -1)
        text = "{}: {:.4f}".format(LABELS[classID], confidence)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)

cv2.imshow("Processed Image", image)
cv2.waitKey(0)
