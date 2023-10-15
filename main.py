import cv2
from PIL import Image

imagePath ="humanFace.jpg"

image = cv2.imread(imagePath)
eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
eyes = eyesCascade.detectMultiScale(image)

# print(eyes)
# for(x, y, w, h) in eyes:
#     cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 0), 3)

face = Image.open(imagePath)
newEyes = Image.open("eye.png")

face = face.convert("RGBA")
newEyes = newEyes.convert(("RGBA"))

for(x, y, w, h) in eyes:
    newEyes = newEyes.resize((w, h))
    face.paste(newEyes, (x, y), newEyes)
    face.save("faceWithNewEyes.png")
    faceWithNewEyesImage = cv2.imread("faceWithNewEyes.png")
    cv2.imshow("faceWithNewEyes", faceWithNewEyesImage)

cv2.imshow("Face", image)
cv2.waitKey()