import cv2
import matplotlib.pyplot as plt
import cvlib as cv
image_path = r'C:\Users\saran\Desktop\kagool\task1clustering\references\profile-cropped.jpeg'
# noinspection PyUnresolvedReferences
im = cv2.imread(image_path)
# noinspection PyUnresolvedReferences
cv2.imshow("image",im)
# noinspection PyUnresolvedReferences
cv2.waitKey(0)
# noinspection PyUnresolvedReferences
cv2.destroyAllWindows()

faces, confidences = cv.detect_face(im)
# loop through detected faces and add bounding box
for face in faces:
    (startX,startY) = face[0],face[1]
    (endX,endY) = face[2],face[3]
    # draw rectangle over face
    # noinspection PyUnresolvedReferences
    cv2.rectangle(im, (startX,startY), (endX,endY), (0,255,0), 2)
# display output
# noinspection PyUnresolvedReferences
img_rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
# noinspection PyUnresolvedReferences
font = cv2.FONT_HERSHEY_SIMPLEX
# org
#co-ordinates of text
org = (100,90)
# fontScale
fontScale = 0.7
# Blue color in BGR
color = (255,255,255)

# Line thickness of 2 px
thickness = 2

# Using cv2.putText() method
# noinspection PyUnresolvedReferences
image = cv2.putText(img_rgb, 'Learner Found! 99.99%', org, font,
                    fontScale, color, thickness, cv2.LINE_AA)
plt.imshow(img_rgb)
plt.show()

