import cv2
import torch
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
import glob
import numpy as np

# YOLOv5 modelini yüklemek için gerekli olan işlemler
model_path = 'yolov5s.pt'  # Eğitilmiş YOLOv5 modelinin yolu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # CUDA kullanılabilirse GPU'yu kullan

model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.to(device).eval()

# Görüntüyü ve boyutunu hesaplamak istediğimiz nesneyi tespit etmek için kullanacağımız görüntü
# image_path = 'C:/Users/Yasin/Desktop/BottleCap/kayit/5.png'  # Tespit yapılacak görüntünün yolu
target_size = (640, 480)
image_path = "D:/SilBastan/open/IMG-20230308-WA0007.jpg"
image = Image.open(image_path)
image = image.resize(target_size, Image.ANTIALIAS) #yeniden boyutlandır.
# YOLOv5 ile nesne tespiti
results = model([image], size=640)  # Görüntüyü YOLOv5'e ver ve sonuçları al

# İlk tespit sonucunu al (varsayılan olarak en güvenilir sonuç)
result = results.pandas().xyxy[0]

# Tespit edilen nesnenin piksel boyutlarını al
x1, y1, x2, y2 = result['xmin'], result['ymin'], result['xmax'], result['ymax']
object_width = x2 - x1
object_height = y2 - y1

"""
*********************Kamera kalibrasyonu****************
"""
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('img/*.png') #kalibrasyon görüntüleri yolu
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None) #tahtanın kenarlarını bul
    
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria) #siyah beyaz geçişlerin olduğu noktaları bul
        imgpoints.append(corners2)
        # Draw and display the corners        
        # cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# camera_matrix = [(588.132,0,320.441),(0,586.565,231.731),(0,0,1)]

# Dönüşüm matrisinin oluşturulması
rotation_matrix = np.zeros(shape=(3, 3))
cv2.Rodrigues(rvecs[0], rotation_matrix)

# Kamera kalibrasyonu ve dönüşüm matrisi kullanarak boyutu gerçek dünya koordinatlarına dönüştür
pixel_size = np.array([[object_width], [object_height], [1]])
real_world_coordinates = np.dot(np.linalg.inv(camera_matrix), np.dot(rotation_matrix, pixel_size))

"""   *****************************  """

print("Gerçek dünya koordinatları:", real_world_coordinates)


# Boyut hesaplama
object_width_mm = abs(real_world_coordinates[0][0])
object_height_mm = abs(real_world_coordinates[1][0])

print("Objenin genişliği (mm):", object_width_mm)
print("Objenin yüksekliği (mm):", object_height_mm)

image = cv2.imread(image_path)
image = cv2.resize(image,(640,480)) #target size ile aynı olmalı
cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Ekranda gösterme
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()







