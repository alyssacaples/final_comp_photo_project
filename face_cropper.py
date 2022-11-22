from mtcnn import MTCNN
from PIL import Image
import cv2


#image = cv2.imread("dataset/taylorswift_1.jpg")

path = "dataset/taylorswift_1.jpg"
# Window name in which image is displayed

#mcnn parser
image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
detector = MTCNN()
result = detector.detect_faces(image)
bounding_box = result[0]['box']
keypoints = result[0]['keypoints']

#crop image based on detector box
pil_im = Image.open(path)
crop_im = pil_im.crop((bounding_box[0], bounding_box[1],bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3] ))

crop_im.show()
  
# Using cv2.imshow() method 
# Displaying the image 
# window_name = 'image'
# cv2.imshow(window_name, crop_im)
  
# #waits for user to press any key 
# #(this is necessary to avoid Python kernel form crashing)
# cv2.waitKey(0) 
  
# #closing all open windows 
# cv2.destroyAllWindows() 

# image = cv2.cvtColor(cv2.imread("dataset/taylorswift_1.jpg"), cv2.COLOR_BGR2RGB)
# detector = MTCNN()
# result = detector.detect_faces(image)

# # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.


# cv2.rectangle(image,
#               (bounding_box[0], bounding_box[1]),
#               (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
#               (0,155,255),
#               2)

# window_name = 'image'
# cv2.imshow(window_name, image)
  
#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
# cv2.waitKey(0) 
  
# #closing all open windows 
# cv2.destroyAllWindows()

# [
#     {
#         'box': [277, 90, 48, 63],
#         'keypoints':
#         {
#             'nose': (303, 131),
#             'mouth_right': (313, 141),
#             'right_eye': (314, 114),
#             'left_eye': (291, 117),
#             'mouth_left': (296, 143)
#         },
#         'confidence': 0.99851983785629272
#     }
# ]