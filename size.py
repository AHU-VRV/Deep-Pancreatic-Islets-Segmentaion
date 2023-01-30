import numpy as np
from skimage import measure, color, morphology,io,filters
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time

start = time.perf_counter()


array_of_properties = []
array_of_area = []
array_of_centroid = []
path = "/media/qing/My Book/swenden_data/segmentation_result/pancreas18/"
files = os.listdir(path)
files.sort(key=lambda x: int(x.split('.')[0]))
num_image = 0
for filename in files:
    print(filename)
    img = cv2.imread(path+filename,cv2.IMREAD_GRAYSCALE)
    bwimg =(img>200)
    dst = morphology.remove_small_objects(bwimg,min_size=1,connectivity=2)
    label = measure.label(dst, connectivity=2)
    properties = measure.regionprops(label)
    temp_area = []
    temp_centroid = []
    for prop in properties:
        temp_area.append(prop.area)
        temp_centroid.append(prop.centroid)
    array_of_area.append(temp_area)
    array_of_centroid.append(temp_centroid)
    num_image = num_image+1



#If the center point coordinates of the white connected areas on two adjacent pictures differ by 5,
# it is considered to be a cross section of the same islet, and the corresponding index is saved to index (k, t, k+1, p)
index=[]
for k in range(num_image-1):
    num = len(array_of_area[k])              #The number of white connected regions in the k-th image
    # pro = array_of_properties[k]
    for t in range(num):
        i = 0
        for p in range(len(array_of_area[k + 1])):
            if abs(array_of_centroid[k + 1][p][0] - array_of_centroid[k][t][0]) <= 5 and abs(array_of_centroid[k + 1][p][1] - array_of_centroid[k][t][1]) <= 5:
                i = i + 1
                if i == 1:
                    index.append((k, t, k + 1, p))


#Integrate the index of the cross section of the same islet
def gather_index(i,a,temp,index):
    while i< len(index):
        if index[i][0] == temp[2] and index[i][1] == temp[3]:
            a.append(index[i])
            #i=i+1
            temp = index[i]
            index.remove(index[i])
        else:
            i = i+1
    return a


index_temp = copy.deepcopy(index)
ind_area=[]
# length_index = len(index_temp)
i = 0
while i < len(index):
    temp = index[i]
    t = [temp]
    a_new = gather_index(i,t,temp, index)
    if len(a_new) >= 1:
        ind_area.append(a_new)
        print(a_new)
    i = i+1


#save the index in .txt file
file=open('save_area/pancreas_18/area_pancreas_index18_2.txt','w')
file.write(str(ind_area));
file.close()
print("Successfully saved")


#Find the cross-sectional area sum
length_ind = len(ind_area)
print('number of islets: ',length_ind)
area_all = []

file=open('save_area/pancreas_18/area_pancreas18_2.txt','w')
for i in range(length_ind):
    #area_temp = array_of_area[]
    temp = ind_area[i]
    length = len(temp)
    area = array_of_area[temp[0][0]][temp[0][1]] + array_of_area[temp[0][2]][temp[0][3]]
    for j in range(1,length):
        area = area + array_of_area[temp[j][2]][temp[j][3]]
    # print(ind_area[i])
    area_all.append(area)
    # print(area_all[i])
    s = str(area).replace('[', '').replace(']', '')
    s = s.replace("'", '').replace(',', '') + '\n'
    file.write(s)
file.close()
print("Successfully save file of size")
print(len(area_all))

end = time.perf_counter()

print("The function run time is : %.03f seconds" %(end-start))
