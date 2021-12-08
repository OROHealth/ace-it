
import pandas as pd
import os

cols=['class','label','img_loc']
df=pd.DataFrame(columns=cols)

arr = os.listdir('D:/pytorch/skin_class/data_3class_skin_diseases/test/acne/')

for i in arr:
    m = 'D:/pytorch/skin_class/data_3class_skin_diseases/test/acne/' + i
    #m = m.replace("/", "\\")
    #print(m)
    df=df.append({'class': 'acne', 'label':0, 'img_loc':m},ignore_index=True)

print(df.head())

arr1= os.listdir('D:/pytorch/skin_class/data_3class_skin_diseases/test/herpes_simplex/')

df1=pd.DataFrame(columns=cols)


for i in arr1:
    m = 'D:/pytorch/skin_class/data_3class_skin_diseases/test/herpes_simplex/' + i
    #m = m.replace("/", "\\")
    #print(m)
    df1=df1.append({'class': 'herpes_simplex', 'label':1, 'img_loc':m},ignore_index=True)


arr2= os.listdir('D:/pytorch/skin_class/data_3class_skin_diseases/test/lichen_planus/')
df2=pd.DataFrame(columns=cols)


for i in arr2:
    m = 'D:/pytorch/skin_class/data_3class_skin_diseases/test/lichen_planus/' + i
    #m = m.replace("/", "\\")
    #print(m)
    df2=df2.append({'class': 'lichen_planus', 'label':2, 'img_loc':m},ignore_index=True)

r,c=df1.shape
print(r,c)
print(df1.head())
frames=[df,df1,df2]
result=pd.concat(frames)
r,c=result.shape
print(r,c)
result.to_csv (r'D:/pytorch/skin_class/data_3class_skin_diseases/data_aug/npy_test_dataset.csv', index = None, header=True)