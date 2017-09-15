import sys
from PIL import Image
westbrook_jpg = Image.open(sys.argv[1])
#Q2_png = open("Q2.png", "w")
#print(westbrook_jpg)
pix = westbrook_jpg.load()
width = westbrook_jpg.size[0]
height = westbrook_jpg.size[1]
for x in range(width):
    for y in range(height):
        r,g,b = pix[x,y]
        pix[x,y] = r//2, g//2, b//2
westbrook_jpg.save("Q2.png")
