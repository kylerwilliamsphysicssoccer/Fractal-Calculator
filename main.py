import numpy as np
import cv2
from PIL import Image as PILImage, ImageTk
import math
import sys 
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from helper import *

#Must import an image of a fractal to your device.

image=PILImage.open("C:/Users/kxia/Desktop/Kyler Coding/Fractal Calculator/tree.gif")
#image=PILImage.open("C:/Users/kxia/Desktop/Kyler Coding/Fractal Calculator/Screenshot 2024-12-08 180026.jpg")
#image=PILImage.open("C:/Users/kxia/Desktop/Kyler Coding/Fractal Calculator/Circle.jpg")
#image=PILImage.open("C:/Users/kxia/Desktop/Kyler Coding/Fractal Calculator/Britain Satellite.jpg")
#image=PILImage.open("C:/Users/kxia/Desktop/Kyler Coding/Fractal Calculator/SierpinskiTriangle.jpg")

matrix1=np.array(image)
#matrix1=np.random.randint(0, 256, size=(1000, 1000))

if(matrix1.ndim>2):
    print("Matrix before summing:", matrix1.shape)
    matrix1=np.sum(matrix1[:, :, -3:], axis=2)
    matrix1=matrix1/3

print("Matrix Dimension: ", str(matrix1.shape))
rank = np.linalg.matrix_rank(matrix1)
print("Matrix Rank: ", str(rank))
matrix=np.copy(matrix1)

#matrix=increaseContrast(matrix)

# dispIm(matrix)

# matrixmeth1= marginalize(matrix, np.median(matrix))
# dispIm(matrixmeth1)

# matrixmeth2= SVD(matrix, 50)
# dispIm(matrixmeth2)
# matrixmeth2= marginalize(matrixmeth2, np.median(matrixmeth2))
# dispIm(matrixmeth2)




counts, second_column = blockSizeCalculator(matrix, 50, 5, -3)

#Ommitted because it does not calculate fractal dimension:
# counts, second_column= rankCalculator(matrix, 100, 5, 1, 5)

first_column = np.ones((second_column.size, 1))
second_column = second_column[:, np.newaxis]
#print("First column shape:", first_column.shape) 
#print("Second column shape:", second_column.shape)
A = np.hstack((first_column, second_column))

#print("A:", A)
#print("A is : ", str(A))
Atrans= A.T
ATA= np.dot(Atrans, A)
ATA_inv = np.linalg.inv(ATA)
ATB= np.dot(Atrans, counts)
x= np.dot(ATA_inv, ATB)
print("Fractal Dimension is Approximately ", -x[1])
slope= x[1]
intercept= x[0]

# sosr = sum of square residuals
sosr= 0
sum=0
for i in range(second_column.size):
    sum+= counts[i]
mean=sum/second_column.size
variance = 0
for i in range(second_column.size):
    ressquare= (counts[i]-(intercept+slope*second_column[i]))**2
    sosr+= ressquare
    variance+=(counts[i]-mean)**2
r_2=1-sosr/variance
print('R square is:', r_2)




    # matrix = np.sum(matrix, axis=2)
    # print("Matrix shape after summing:", matrix.shape)

    # matrix = np.where(matrix > 255, 255, matrix)
    # tensor = np.zeros((matrix.shape[0], matrix.shape[1], 3))
    # tensor[:,:,0]= matrix
    # display = matrix

    

# image=PILImage.open("C:/Users/kxia/Desktop/Kyler Coding/Fractal Calculator/tree.gif")
# img_width, img_height = image.size 
# imageLength=img_width
# imageWidth=img_height
# image = image.resize((math.floor(imageLength/1), math.floor(imageWidth/1)), PILImage.Resampling.LANCZOS)
# matrix=np.array(image)
# print(str(matrix.shape))
# #print(str(matrix[10][10][0]))
# #matrix is 1220 by 1704

# img_width, img_height = image.size 
# imageLength=img_width
# imageWidth=img_height
# print(f"Image dimensions: {img_width}x{img_height}")
# if img_width != imageLength or img_height != imageWidth: 
#     print("Image dimensions do not match window dimensions")

# # image = ImageTk.PhotoImage(image)
# with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as temp_file: 
#     temp_image_path = temp_file.name
#     image.save(temp_image_path)

# #windowSize
# winWidth=1704
# winHeight=1220

# #Length of squares
# length= 100
# numsquares=[]
# while(length>5):
#     win= GraphWin("Fractals!", winWidth, winHeight)
#     fractalImage=Image(Point(math.floor(winWidth/2), math.floor(winHeight/2)), temp_image_path)
#     fractalImage.draw(win)
#     win.getMouse()
#     win.close()
#     length=math.floor(length/1.5)
