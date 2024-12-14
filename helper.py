import numpy as np
import cv2
from PIL import Image as PILImage, ImageTk
import math
import sys 
import tempfile
#sys.path.append('C:/Users/kxia/Desktop/Kyler Coding/Fractal Calculator/graphics.py') # Replace with actual path
import matplotlib.pyplot as plt
import numpy as np




def marginalize(mat, cutoff):
    mat= np.where(mat > cutoff, 255, mat)
    mat= np.where(mat < cutoff, 0, mat)
    return mat

def dispIm(matrixDis):
    plt.imshow(matrixDis, cmap='gray')
    plt.axis('off')
    plt.show()

def SVD(matrix2, num):
    U, S, VT = np.linalg.svd(matrix2)
    #print("U: ", U, "done")
    U_num=U[:, :num]
    S_num=np.diag(S[:num])
    VT_num=VT[:num, :]
    return np.dot(U_num, np.dot(S_num, VT_num))
    #matrix=np.dot(matrix.T, matrix)

def blockSizeCalculator(matrix, largestSize, stepSize, blockSize):
    x=matrix.shape[0]
    y=matrix.shape[1]
    second_column =np.array([])
    counts =np.array([])
    while(blockSize<largestSize):
        blockSize+=stepSize
        display=np.copy(matrix)
        count=0
        for i in range(math.floor(x/blockSize)):
            for j in range(math.floor(y/blockSize)):
                initx=i*blockSize
                inity=j*blockSize
                contains=False
                for k in range(blockSize-1):
                    if(contains):
                        break
                    for l in range (blockSize-1):
                        if(contains):
                            break
                        if(matrix[initx+k][inity+l]<100):
                            contains=True
                if(contains):
                    count+=1
                    for k in range(blockSize-1):
                        for l in range (blockSize-1):
                            display[initx+k][inity+l]-=100
                        
        counts=np.append(counts, np.log(count))
        second_column=np.append(second_column, np.log(blockSize))
        print("Block Size: ", blockSize, ". Count: ", count, "Ln Count: ", np.log(count))
        display= np.where(display > 255, 255, display)
        display= np.where(display <0, 0, display)

        dispIm(display)

    return counts, second_column


def rankCalculator(matrix, largestRank, stepSize, rankSize, blockSize):
    x=matrix.shape[0]
    y=matrix.shape[1]
    second_column =np.array([])
    counts =np.array([])

    while(rankSize<largestRank):
        rankSize+=stepSize
        display=np.copy(matrix)
        display=SVD(display, rankSize)

        count=0
        for i in range(math.floor(x/blockSize)):
            for j in range(math.floor(y/blockSize)):
                initx=i*blockSize
                inity=j*blockSize
                contains=False
                for k in range(blockSize-1):
                    if(contains):
                        break
                    for l in range (blockSize-1):
                        if(contains):
                            break
                        if(display[initx+k][inity+l]<100):
                            contains=True
                if(contains):
                    count+=1
                    for k in range(blockSize-1):
                        for l in range (blockSize-1):
                            display[initx+k][inity+l]-=100
                        
        counts=np.append(counts, np.log(count))
        second_column=np.append(second_column, np.log(blockSize))
        print("SVD Rank: ", rankSize, ". Count: ", count, "Ln Count: ", np.log(count))
        display= np.where(display > 255, 255, display)
        display= np.where(display <0, 0, display)
        
        dispIm(display)

    return counts, second_column


def increaseContrast(matrix):   
    fir=matrix.shape[0]
    sec=matrix.shape[1]
  
    for a in range(5):
        disp=np.copy(matrix)
        for i in range(1, fir-1):
            for j in range(1, sec-1):
                greater=False
                less=False
                for k in range (3):
                    k1=i+k-1
                    for l in range (3):
                        l1=j+l-1
                        if(matrix[i][j]-matrix[k1][l1]>10):
                            greater=True
                        if(matrix[i][j]-matrix[k1][l1]<-10):
                            less=True
                if(greater):
                    disp[i][j]+=50
                if(less):
                    disp[i][j]-=50
        matrix=disp
        matrix = np.where(matrix > 255, 255, matrix)
        matrix = np.where(matrix <0, 0, matrix)
        dispIm(matrix)

    return matrix