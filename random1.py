import cv2

img = cv2.imread(r"C:\Users\KIIT\Desktop\medical-image-super-resolution\results\yy.png")
if img is None:
    print("Failed to load image.")
else:
    print("Image loaded successfully.")
