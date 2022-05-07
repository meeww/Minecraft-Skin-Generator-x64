#!C:/Users/joejo/AppData/Local/Programs/Python/Python38/python.exe
from PIL import Image

img = Image.open("results/generated.png")

width,height = img.size


top = 2
bottom = 2

for i in range(4):
    left = 66*i +2
    right = width-66*i
    imgcrop = img.crop((left,top,left+64,66))

    imgcrop.save("results/"+str(i)+".png")