from mtcnn import MTCNN
import cv2
import os, time
from numpy import asarray
from PIL import Image

# def crop(img, results, face):
#     x1, y1, width, height = results[face]["box"]
#     x2, y2 = x1+width, y1+height
#     cropped_face = img[y1:y2, x1:x2]
#     face_image = Image.fromarray(cropped_face)
#     face_image = face_image.resize((224,224))
#     #face_array = asarray(face_image)
#     return face_image

# def face_crop_in_image(faces_image_path, faces_folder_path):
#     img = cv2.imread(faces_image_path)
#     results = detector.detect_faces(img)
#     face_image = crop(img, results, 0)
#     cv2.imwrite(f"faces_folder_path.jpg", cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
#     print(results)

def crop(img, results, face):
    x1, y1, width, height = results[face]["box"]
    width2 = round(width * 1.6)
    height2 = round(height * 1.6)
    
    x1 = round(x1 - (width2/6))
    y1 = round(y1 - (height2/6))
    x2, y2 = x1+width2, y1+height2
    cropped_face = img[y1:y2, x1:x2]
    face_image = Image.fromarray(cropped_face)
    face_image = face_image.resize((224,224))
    face_array = asarray(face_image)
    return face_array

def face_crop_in_image(faces_image_path, faces_folder_path):
    img = cv2.imread(faces_image_path)
    results = detector.detect_faces(img)
    print("[PLEASE WAIT]")
    start_time = time.time()
    for i in range(len(results[0:])):
        face_array = crop(img, results, i)
        cv2.imwrite("faces_folder_path.jpg", face_array)
    print(f"[FINISHED] [TIME: {time.time() - start_time :.2f} seconds]")

image = cv2.cvtColor(cv2.imread("matthew.jpg"), cv2.COLOR_BGR2RGB)

detector = MTCNN()
result = detector.detect_faces(image)

bounding_box = result[0]['box']
keypoints = result[0]['keypoints']

cv2.rectangle(image,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255),
              2)

cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

cv2.imwrite("matthew_draw.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

face_crop_in_image("taylorswift_1.jpg", "matthew_draw2.jpg")

print(result)


