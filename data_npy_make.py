import pandas as pd
import  numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
df = pd.read_csv('D:/pytorch/skin_class/data_3class_skin_diseases/data_aug/npy_test_dataset.csv')
#df=pd.read_excel('D:/polynomial/covid19/anomaly dataset/test_train.xlsx')
#print(df)
df=shuffle(df)
print(df.head())


reshaped_image = df["img_loc"].map(lambda x: np.asarray(Image.open(x).resize((224,224), resample=Image.LANCZOS).\
                                                          convert("RGB")))

out_vec = np.stack(reshaped_image, 0)

print(out_vec.shape)

out_vec = out_vec.astype("float32")
print(out_vec.max())
out_vec /= 255.0

labels = df["label"].values
print(labels)
print(out_vec.shape)

'''X_train, X_val, y_train, y_val = train_test_split(out_vec, labels, test_size=0.10,  stratify=labels)

np.save("D:/pytorch/skin_class/data_3class_skin_diseases/skin_224_224_val.npy", X_val)
np.save("D:/pytorch/skin_class/data_3class_skin_diseases/skin_val_labels.npy", y_val)

np.save("D:/pytorch/skin_class/data_3class_skin_diseases/skin_224_224_train.npy", X_train)
np.save("D:/pytorch/skin_class/data_3class_skin_diseases/skin_train_labels.npy", y_train)'''


np.save("D:/pytorch/skin_class/data_3class_skin_diseases/test.npy", out_vec)
np.save("D:/pytorch/skin_class/data_3class_skin_diseases/test_labels.npy", labels)

