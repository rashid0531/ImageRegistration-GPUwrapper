
import os

for i in range(62,65):
    image= "%04d" % i
    print(image)
list=[]
impath= "/home/mrc689/ImageRegistration/Data/drone_images_tif/drone_images_tif_1/"
for f in os.listdir(impath):
    if os.path.isfile(os.path.join(impath, f)):
       list.append(f)

list.sort()
print(list)






