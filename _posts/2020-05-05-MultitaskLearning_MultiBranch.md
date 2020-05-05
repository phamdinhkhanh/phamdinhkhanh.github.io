---
layout: post
author: phamdinhkhanh
title: Bài 35 - Multitask Learning - Multi Branch
---

# 1. Giới thiệu chung.

ở bài trước chúng ta đã tìm hiểu về học đa tác vụ ([multitask learning](https://phamdinhkhanh.github.io/2020/04/22/MultitaskLearning.html)). Thuật toán có khả năng dự báo nhiều tác vụ trên cùng một đầu vào, có sự chia sẻ đặc trưng giữa các tác vụ trong quá trình dự báo mà không cần xây dựng nhiều mô hình độc lập. Tiết kiệm được tài nguyên và thời gian huấn luyện. 

Multitask learning thường được ứng dụng trong nhiều lĩnh vực trong đó có xe tự hành, image search, sinh trắc học trên khuôn mặt, y sinh và rất nhiều các ứng dụng khác. Bạn đọc có thể tham khảo thêm về ứng dụng của multitask learning trong sinh trắc học khuôn mặt và chuẩn đoán ảnh X-quang trong hội nghị toàn cầu về xử lý ảnh [ICCV 2019](https://www.youtube.com/watch?v=9Sx2qWKGzlc&list=PLBAFw2oKatLJ7Mb8Xat6fq_oGyD7JdaNU&index=4) (từ phút thứ 60 của video) ứng dụng trong sinh trắc học trên khuôn mặt và y sinh.

Chúng ta cũng được tìm hiểu ở bài trước rằng multitask learning sẽ tương tốt trong trường hợp quá trình phân biệt các tác vụ chia sẻ nhiều đặc trưng giống nhau. Vậy đối với trừng hợp không chia sẻ đặc trưng thì multitask learning sẽ cần điều chỉnh như thế nào? Đó chính là kiến trúc rẽ nhánh mà chúng ta sẽ cùng tìm hiểu ở bài này.



# 2. Kiến trúc rẽ nhánh (multi-branch)



Kiến trúc rẽ nhánh sẽ cho phép thuật toán học được nhiều tác vụ đồng thời nhưng không chia sẻ đặc trưng. Mô hình của chúng ta sử dụng chung một đầu vào là ảnh và phân nhánh thành nhiều mô hình con. Mỗi mô hình sẽ phụ trách dự báo cho một tác vụ một cách độc lập.



Ví dụ: Trong nhận diện khuôn mặt, chúng ta sẽ cần sử dụng rất nhiều các dự báo trên cùng một ảnh khuôn mặt như: giới tính, độ tuổi, chủng tộc, màu mắt, màu tóc,....

Những tác vụ trên không chia sẻ các đặc trưng để phân biệt. Ví dụ: Khi phân biệt giới tính chúng ta dựa trên các đặc trưng về độ dài tóc, râu, lông mày, mắt, cằm và quai hàm nhiều hơn nhưng phân biệt độ tuổi chúng ta chủ yếu dựa vào nết nhăn trên khuôn mặt, màu da, màu tóc. Đây là những đặc trưng không hoàn toàn giống nhau. Do đó sử dụng kiến trúc multitask learning chia sẻ tham số cho bài toán này sẽ không hợp lý.

Một lựa chọn tốt hơn trong trường hợp này cho chúng ta đó là xây dựng một kiến trúc rẽ nhánh ngay từ input layer. Giữa các nhánh là độc lập, chỉ sử dụng chung một đầu vào mà không chia sẻ tham số.

<img src="/assets/images/20200505_MultitaskLearningMultiBranch/pic1.png" class="largepic"/>

**Hình 1:** Kiến trúc rẽ nhánh của mô hình multitask learning trong tác vụ dự báo `age, gender, race`. Mỗi nhánh thực hiện một tác vụ như sau:

* Nhánh thứ 1: Dự báo độ tuổi dựa trên thuật toán regression.
* Nhánh thứ 2: Là một bài toán phân loại nhị phân (binary classification) dự báo giới tính.
* Nhánh thứ 3: Là một bài toán phân loại đa lớp (multiclass classification) dự báo 5 chủng tộc khác nhau của khuôn mặt.


Hàm loss function tổng quát cần tối ưu vẫn là tổng có trọng số của toàn bộ loss function trên từng nhánh model.

# 3. Ưu và nhược điểm của kiến trúc rẽ nhánh

## 3.1. Ưu điểm của kiến trúc rẽ nhánh

* Kiến trúc rẽ nhánh sẽ cho phép xây dựng được nhiều mô hình trên cùng một bộ dữ liệu.

* Các mô hình là những tác vụ độc lập và không có các đặc trưng phân loại giống nhau. 

* Khi một tác vụ hoạt động không tốt thì cũng không ảnh hưởng tới những tác vụ khác vì mối quan hệ giữa các mô hình là độc lập.

* Tiết kiệm tài nguyên và chi phí thời gian tính toán vì chúng ta huấn luyện các tác vụ đồng thời trên cùng một mô hình mà không cần huấn luyện lại từng mô hình đơn lẻ.

* Kiến trúc rẽ nhánh đồng thời cũng kết hợp được giữa các bài toán classifcation và prediction với nhau trong cùng một kiến trúc mô hình.

* Kiến trúc rẽ nhánh cho phép ta áp dụng được nhiều hàm loss function khác nhau trên các tác vụ huấn luyện khác nhau.


## 3.2. Nhược điểm của kiến trúc rẽ nhánh

* Kiến trúc rẽ nhánh thường yêu cầu các mô hình phải có input image cùng một shape. Một số tác vụ dự báo chỉ cần một shape nhỏ hơn cũng đã cho kết quả tốt nên sẽ gây lãng phí tài nguyên tính toán nếu tất cả các mô hình sử dụng chung một high resolution image.

* Loss function của kiến trúc rẽ nhánh rất đa dạng, trong đó một số tác vụ có thể có loss function đóng góp đa số vào loss function tổng quát cuối cùng. Do đó cần thiết lập trọng số cho loss function để cân bằng ảnh hưởng của chi phí mất mát giữa các tác vụ.

Để hiểu rõ hơn ưu và nhược điểm của kiến trúc rẽ nhánh, chúng ta sẽ cùng đi vào phần thực hành.



# 4. Thực hành xây dựng mô hình multitask learning

## 4.1. Dataset

Bộ dữ liệu về face được sử dụng là [UTKFace](https://susanqq.github.io/UTKFace/). Bộ dữ liệu bao gồm ảnh của đa dạng các sắc tộc, giới tính, độ tuổi.

Tên các files ảnh được đặt theo cấu trúc: `[age]_[gender]_[race]_[date&time].jpg` Trong đó:

* [age]: độ tuổi, giá trị là một số nguyên từ 0 đến 116.
* [gender]: giới tính.
* [race]: chủng tộc, giá trị là số nguyên từ 0-4 lần lượt tương ứng với 5 sắc tộc da trắng, da đen, châu á, ấn độ và sắc tộc khác (như người tây ban nha, châu mỹ latin, trung đông).
* [date&time]: ngày tháng chụp ảnh có định dạng yyyymmddHHMMSSFFF.


Để thực hành thì các bạn chỉ cần download file [crop_part1.tar.gz](https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE) là những ảnh khuôn mặt đã được crop bằng cách thêm lỗi tắt vào google drive. Nhớ sắp xếp chúng vào cùng thư mục với file notebook.

<img src="/assets/images/20200505_MultitaskLearningMultiBranch/pic2.png" class="largepic"/>

Bạn đọc mở file sau đây để thực hành [Multitask learning - Multi branch Google Colab](https://colab.research.google.com/drive/1hmZxj3s9KScWUCbAi3MhlWtQL9XxzJdxhttps://colab.research.google.com/drive/1hmZxj3s9KScWUCbAi3MhlWtQL9XxzJdx) hoặc thực hiện tuần tự theo các step ở file này.


Mount google drive đến folder chứa colab notebook file.






```
from google.colab import drive
import os

drive.mount('/content/gdrive')
path = '/content/gdrive/My Drive/Colab Notebooks/MultitaskLearning'
os.chdir(path)
!ls
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly
    
    Enter your authorization code:
    ··········
    Mounted at /content/gdrive
     Bai34MultitaskLearning        MultitaskLearning.ipynb
     checkpoint		       MultitaskLearning_MultiBranchOutput.ipynb
     crop_part1		      'UTKface_Aligned&cropped'
     multi-output-classification   UTKface_crop_part1
     multitaskLearning
    

Sau đó chúng ta giải nén `UTKface_crop_part1/crop_part1.tar.gz` bằng lện tar.


```
!tar -xvf UTKface_Aligned\&cropped/crop_part1.tar.gz
```

Tiếp theo chúng ta sẽ trích lọc các nhãn từ link ảnh bao gồm: `age, gender, race`.


```
import glob2
imagePaths = glob2.glob('crop_part1/*')
print('Number of images', len(imagePaths))

# Extract age, gender, race từ link ảnh

multiLabels = []
ages = []
genders = []
races = []
linkImages = []

for i, imageLink in enumerate(imagePaths):
  try:
    age, gender, race, _ = imageLink.split('/')[1].split('_')
    linkImages.append(imageLink)
  except:
    next
  age = int(age)
  gender = int(gender)
  race = int(race)
  multiLabels.append((age, gender, race))
  ages = [label[0] for label in multiLabels]
  genders = [label[1] for label in multiLabels]
  races = [label[2] for label in multiLabels]

dict_labels = {'ages': ages,
               'genders': genders,
               'races': races,
               'linkImages': linkImages}
```

    Number of images 0
    


```
import matplotlib.pyplot as plt

X = plt.imread('crop_part1/27_1_3_20170104232751618.jpg.chip.jpg')
plt.imshow(X)
```




    <matplotlib.image.AxesImage at 0x7f50b1de2b00>




![png](MultitaskLearning_MultiBranchOutput_Face_files/MultitaskLearning_MultiBranchOutput_Face_12_1.png)


Tiếp theo chúng ta sẽ kiểm tra phân phối giữa các nhóm trong cùng một biến nhằm phát hiện những bất thường về dữ liệu như hiện tượng mất cân đối nghiêm trọng giữa các nhóm.


```
import matplotlib.pyplot as plt
import numpy as np

def _plot_bar(varname):
  labels, values = np.unique(dict_labels[varname], return_counts = True)
  plt.figure(figsize=(10, 6))
  ax = plt.subplot()
  ax.bar(x=labels, height=values)
  ax.set_title('No samples of ' + varname)
  ax.set_xlabel(varname)
  ax.set_ylabel('No samples')
  plt.show()

_plot_bar('ages')
_plot_bar('genders')
_plot_bar('races')
```


![png](MultitaskLearning_MultiBranchOutput_Face_files/MultitaskLearning_MultiBranchOutput_Face_14_0.png)



![png](MultitaskLearning_MultiBranchOutput_Face_files/MultitaskLearning_MultiBranchOutput_Face_14_1.png)



![png](MultitaskLearning_MultiBranchOutput_Face_files/MultitaskLearning_MultiBranchOutput_Face_14_2.png)


Nhìn vào số liệu ta có thể thấy:

* Bộ dữ liệu có nhiều ảnh của trẻ sơ sinh và người từ 20-30 tuổi. Tỷ lệ người già chiếm một phần thiểu số.
* Tỷ lệ người da trắng chiếm đa số trong bộ dữ liệu (khoảng 5000 ảnh và trên 50%), ít nhất là người da đen (chỉ khoảng 3-5%) và các chủng tộc còn lại chiếm tỷ lệ giao động từ 10-15%.
* Có hiện tượng mất cân bằng giới tính, Giới tính nữ (nhãn 1) nhiềm hơn nam (nhãn 0).

Trong trường hợp này chúng ta có thể thực hiện một số biện pháp data augumentation để tăng cường thêm ảnh cho những nhóm thiểu số. Tuy nhiên với mục đích chính là hiểu rõ thuật toán mình sẽ bỏ qua bước này.

## 4.2. Mô hình

Mỗi một lượt huấn luyện, toàn bộ các mô hình con sẽ có chung một ảnh đầu vào. Sau đó từ đầu vào sẽ rẽ nhánh tới từng mô hình con phục vụ cho các mục tiêu dự báo khác nhau về `age, gender, race`. Mỗi một mô hình con sẽ sử dụng một mạng Convolutional Neural Network với cùng kích thước inputshape là ảnh `96 x 96 x 3`. 

**Thêm ảnh về kiến trúc các nhánh mô hình**

Các hàm `_age_basenetwork()`, `_gender_basenetwork()`, `_race_basenetwork()` sẽ có tác dụng khởi tạo những mô hình con CNN tương ứng với các tác vụ phân loại `age, gender` và `race`.


```
%tensorflow_version 2.x

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, DepthwiseConv2D, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

INPUT_SHAPE = (96, 96, 3)
N_AGES = 1
N_GENDERS = 1
N_RACES = 5

class AgeGenderRaceNet(object):
  def _age_basenetwork(inputs, classes = N_AGES, finAct = 'linear'):
    # DepthWiseCONV => CONV => RELU => POOL
    x = DepthwiseConv2D(kernel_size=(3, 3), 
                        padding="same", activation='relu')(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)

    # (CONV => RELU) * 4 => POOL
    x = Conv2D(filters=64, kernel_size=(3, 3), padding="same",
          activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding="same",
          activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding="same",
          activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=1024, kernel_size=(3, 3), padding="same",
          activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # first (and only) set of FC => RELU layers
    x = Flatten()(x)
    x = Activation("relu")(x)

    # softmax classifier
    x = Dense(classes)(x)
    x = Activation(finAct, name="age_output")(x)

    return x
    
  def _gender_basenetwork(inputs, classes = N_GENDERS, finAct = 'sigmoid'):
    # CONV => RELU => POOL
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)

    # CONV => RELU => POOL
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # CONV => RELU => POOL
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # MLP classifier
    # Số units của output phải bằng số lượng nhóm giới tính
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dense(classes)(x)
    x = Activation(finAct, name="gender_output")(x)

    return x

  def _race_basenetwork(inputs, classes = N_RACES, finAct = 'softmax'):
    # CONV => RELU => POOL
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)

    # CONV => RELU => POOL
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # CONV => RELU => POOL
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # MLP classifier
    # Số units của output phải bằng số lượng nhóm sắc tộc
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dense(classes)(x)
    x = Activation(finAct, name="race_output")(x)

    return x

  @staticmethod
  def build(inputShape=INPUT_SHAPE, numGender = N_GENDERS, numRace = N_RACES):
    inputs = Input(shape=INPUT_SHAPE)
    ageBranch=AgeGenderRaceNet._age_basenetwork(inputs=inputs,
      classes=1, finAct='linear')
    genderBranch=AgeGenderRaceNet._gender_basenetwork(inputs=inputs,
      classes = numGender, finAct='sigmoid')
    raceBranch=AgeGenderRaceNet._race_basenetwork(inputs=inputs,
      classes = numRace, finAct='softmax')

    # Tạo một mô hình sử dụng đầu vào là một batch images, sau đó mô hình sẽ 
    # rẽ nhánh, một nhánh xác định đặc trưng của colors và một nhánh xác định đặc trưng của fashion
    model = Model(
      inputs=inputs,
      outputs=[ageBranch, genderBranch, raceBranch],
      name="age_gender_race_net")

    return model

model = AgeGenderRaceNet.build(inputShape=INPUT_SHAPE, numGender = N_GENDERS, numRace = N_RACES)
model.summary()
```

    Model: "age_gender_race_net"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_2 (InputLayer)            [(None, 96, 96, 3)]  0                                            
    __________________________________________________________________________________________________
    depthwise_conv2d_1 (DepthwiseCo (None, 96, 96, 3)    30          input_2[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization_14 (BatchNo (None, 96, 96, 3)    12          depthwise_conv2d_1[0][0]         
    __________________________________________________________________________________________________
    conv2d_11 (Conv2D)              (None, 96, 96, 32)   896         batch_normalization_14[0][0]     
    __________________________________________________________________________________________________
    batch_normalization_15 (BatchNo (None, 96, 96, 32)   128         conv2d_11[0][0]                  
    __________________________________________________________________________________________________
    conv2d_16 (Conv2D)              (None, 96, 96, 16)   448         input_2[0][0]                    
    __________________________________________________________________________________________________
    conv2d_19 (Conv2D)              (None, 96, 96, 16)   448         input_2[0][0]                    
    __________________________________________________________________________________________________
    max_pooling2d_8 (MaxPooling2D)  (None, 32, 32, 32)   0           batch_normalization_15[0][0]     
    __________________________________________________________________________________________________
    batch_normalization_20 (BatchNo (None, 96, 96, 16)   64          conv2d_16[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_24 (BatchNo (None, 96, 96, 16)   64          conv2d_19[0][0]                  
    __________________________________________________________________________________________________
    dropout_5 (Dropout)             (None, 32, 32, 32)   0           max_pooling2d_8[0][0]            
    __________________________________________________________________________________________________
    max_pooling2d_10 (MaxPooling2D) (None, 32, 32, 16)   0           batch_normalization_20[0][0]     
    __________________________________________________________________________________________________
    max_pooling2d_13 (MaxPooling2D) (None, 32, 32, 16)   0           batch_normalization_24[0][0]     
    __________________________________________________________________________________________________
    conv2d_12 (Conv2D)              (None, 32, 32, 64)   18496       dropout_5[0][0]                  
    __________________________________________________________________________________________________
    dropout_6 (Dropout)             (None, 32, 32, 16)   0           max_pooling2d_10[0][0]           
    __________________________________________________________________________________________________
    dropout_8 (Dropout)             (None, 32, 32, 16)   0           max_pooling2d_13[0][0]           
    __________________________________________________________________________________________________
    batch_normalization_16 (BatchNo (None, 32, 32, 64)   256         conv2d_12[0][0]                  
    __________________________________________________________________________________________________
    conv2d_17 (Conv2D)              (None, 32, 32, 32)   4640        dropout_6[0][0]                  
    __________________________________________________________________________________________________
    conv2d_20 (Conv2D)              (None, 32, 32, 32)   4640        dropout_8[0][0]                  
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, 32, 32, 128)  73856       batch_normalization_16[0][0]     
    __________________________________________________________________________________________________
    batch_normalization_21 (BatchNo (None, 32, 32, 32)   128         conv2d_17[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_25 (BatchNo (None, 32, 32, 32)   128         conv2d_20[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_17 (BatchNo (None, 32, 32, 128)  512         conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    max_pooling2d_11 (MaxPooling2D) (None, 16, 16, 32)   0           batch_normalization_21[0][0]     
    __________________________________________________________________________________________________
    max_pooling2d_14 (MaxPooling2D) (None, 16, 16, 32)   0           batch_normalization_25[0][0]     
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, 32, 32, 256)  295168      batch_normalization_17[0][0]     
    __________________________________________________________________________________________________
    dropout_7 (Dropout)             (None, 16, 16, 32)   0           max_pooling2d_11[0][0]           
    __________________________________________________________________________________________________
    dropout_9 (Dropout)             (None, 16, 16, 32)   0           max_pooling2d_14[0][0]           
    __________________________________________________________________________________________________
    batch_normalization_18 (BatchNo (None, 32, 32, 256)  1024        conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    conv2d_18 (Conv2D)              (None, 16, 16, 32)   9248        dropout_7[0][0]                  
    __________________________________________________________________________________________________
    conv2d_21 (Conv2D)              (None, 16, 16, 32)   9248        dropout_9[0][0]                  
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, 32, 32, 1024) 2360320     batch_normalization_18[0][0]     
    __________________________________________________________________________________________________
    batch_normalization_22 (BatchNo (None, 16, 16, 32)   128         conv2d_18[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_26 (BatchNo (None, 16, 16, 32)   128         conv2d_21[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_19 (BatchNo (None, 32, 32, 1024) 4096        conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    max_pooling2d_12 (MaxPooling2D) (None, 8, 8, 32)     0           batch_normalization_22[0][0]     
    __________________________________________________________________________________________________
    max_pooling2d_15 (MaxPooling2D) (None, 8, 8, 32)     0           batch_normalization_26[0][0]     
    __________________________________________________________________________________________________
    max_pooling2d_9 (MaxPooling2D)  (None, 16, 16, 1024) 0           batch_normalization_19[0][0]     
    __________________________________________________________________________________________________
    flatten_4 (Flatten)             (None, 2048)         0           max_pooling2d_12[0][0]           
    __________________________________________________________________________________________________
    flatten_5 (Flatten)             (None, 2048)         0           max_pooling2d_15[0][0]           
    __________________________________________________________________________________________________
    flatten_3 (Flatten)             (None, 262144)       0           max_pooling2d_9[0][0]            
    __________________________________________________________________________________________________
    dense_6 (Dense)                 (None, 128)          262272      flatten_4[0][0]                  
    __________________________________________________________________________________________________
    dense_8 (Dense)                 (None, 128)          262272      flatten_5[0][0]                  
    __________________________________________________________________________________________________
    activation_1 (Activation)       (None, 262144)       0           flatten_3[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_23 (BatchNo (None, 128)          512         dense_6[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization_27 (BatchNo (None, 128)          512         dense_8[0][0]                    
    __________________________________________________________________________________________________
    dense_5 (Dense)                 (None, 1)            262145      activation_1[0][0]               
    __________________________________________________________________________________________________
    dense_7 (Dense)                 (None, 1)            129         batch_normalization_23[0][0]     
    __________________________________________________________________________________________________
    dense_9 (Dense)                 (None, 5)            645         batch_normalization_27[0][0]     
    __________________________________________________________________________________________________
    age_output (Activation)         (None, 1)            0           dense_5[0][0]                    
    __________________________________________________________________________________________________
    gender_output (Activation)      (None, 1)            0           dense_7[0][0]                    
    __________________________________________________________________________________________________
    race_output (Activation)        (None, 5)            0           dense_9[0][0]                    
    ==================================================================================================
    Total params: 3,572,593
    Trainable params: 3,568,747
    Non-trainable params: 3,846
    __________________________________________________________________________________________________
    

Mô hình dự đoán tuổi sẽ có nhiều classes hơn, do đó kiến trúc mạng được thiết kế sẽ sâu hơn so với mô hinh dự đoán chủng tộc và giới tính.

Kiến trúc của các mô hình con đều rất đơn giản. Đó là các layer `[CONV*m-MAXPOOLING*n]*p` liên tiếp nhau.

## 4.3. Huấn luyện mô hình

Để huấn luyện mô hình mình sẽ sử dụng ImageGenerator, bạn đọc xem lại [Bài 32 - Kĩ thuật tensorflow Dataset](https://phamdinhkhanh.github.io/2020/04/09/TensorflowDataset.html#321-s%E1%BB%AD-d%E1%BB%A5ng-imagegenerator) để hiểu thêm về ImageGenerator.


```
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_aug = ImageDataGenerator(rotation_range=25, 
                         width_shift_range=0.1, height_shift_range=0.1, 
                         shear_range=0.2, zoom_range=0.2,
	                       horizontal_flip=True, fill_mode="nearest")
```

Optimizer mình sử dụng là Adam với learning rate được khởi tạo là 0.005.


```
LR_RATE = 0.005
EPOCHS = 50
opt = Adam(lr=LR_RATE, decay=LR_RATE / EPOCHS)
```

Đối với hai bài toán phân loại gender và race chúng ta sử dụng hàm loss function dạng cross entropy. Giả sử bài toán phân loại có $C$ classes output. $y_{i}$ là giá trị thực tế của output và $\hat{y}_{i}$ là giá trị dự báo từ mô hình. Khi đó:
 

$$\mathcal{L}(\mathbf{W}; \mathbf{X}) = -\sum_{i=1}^{N}\sum_{j=1}^{C} y_{ij}.\log(\hat{y}_{ij})$$

Tuy nhiên đối với Age có thể coi là một biến liên tục, thay vì xây dựng một mô hình classification thì chúng ta sẽ coi đây là một bài toán dự báo với biến liên tục. Hàm loss function được sử dụng sẽ là MSE:

$$\mathcal{L}(\mathbf{W}; \mathbf{X}) = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_{i} - y_{i})^2$$

Loss function tổng quát sẽ bằng tổng loss function ở cả 3 nhánh.

Hàm MSE sẽ có trị số lớn gấp nhiều lần so với cross entropy nên để giảm ảnh hưởng của MSE ta sẽ thiết lập trọng số loss function của age nhỏ hơn so với gender và race.


```
losses = {
	"age_output": "mean_squared_error",
	"gender_output": "binary_crossentropy",
	"race_output": "sparse_categorical_crossentropy"
}

lossWeights = {"age_output": 0.03, "gender_output": 1.0, "race_output": 1.0}
model.compile(loss=losses, loss_weights= lossWeights, optimizer=opt, metrics=['accuracy'])
```

### 4.3.1. Phân chia tập train/validation

Như thường lệ chúng ta sẽ phân chia tập train và validation theo tỷ lệ 80/20. Tập train cho huấn luyện và tập validation cho kiểm định.



```
import cv2
import numpy as np

# Phân chia tập train/validation theo tỷ lệ 80/20
n_samples = len(dict_labels['linkImages'])
indices = np.arange(n_samples)
np.random.shuffle(indices)
n_train = int(0.8*n_samples)
index_train, index_val = indices[:n_train], indices[n_train:]
image_train, image_val = [], []
gender_train, gender_val = [], []
race_train, race_val = [], []
label_train, label_val = [], []

def _image_path(index_train):
  image_train = []
  label_train = []
  age_train = []
  gender_train = []
  race_train = []
  for i in index_train:
    imagePath = dict_labels['linkImages'][i]
    # Đọc dữ liệu ảnh
    image = cv2.imread(imagePath)
    image = cv2.resize(image, INPUT_SHAPE[:2])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    image_train.append(image)
    # Gán nhãn cho age, gender, race
    age_train.append(dict_labels['ages'][i])
    gender_train.append(dict_labels['genders'][i])
    race_train.append(dict_labels['races'][i])
    label_train.append([dict_labels['ages'][i], dict_labels['genders'][i], dict_labels['races'][i]])
  
  # Stack list numpy array của ảnh thành một array
  image_train = np.stack(image_train)
  # label_train = np.stack(label_train)
  age_train = np.stack(age_train)
  gender_train = np.stack(gender_train)
  race_train = np.stack(race_train)
  image_train = image_train/255.0
  return image_train, label_train, age_train, gender_train, race_train

image_train, label_train, age_train, gender_train, race_train = _image_path(index_train)
image_val, label_val, age_val, gender_val, race_val = _image_path(index_val)
```

### 4.3.2. Huấn luyện mô hình

Chúng ta sẽ huấn luyện mô hình qua 50 epochs. Sử dụng checkpoint để lưu model có loss function nhỏ nhất trên validation và đồng thời backup mô hình sau mỗi 20 epochs.


```
from tensorflow.keras.callbacks import ModelCheckpoint
import os

if not os.path.exists('checkpoint'):
	os.mkdir('checkpoint')


cp_callback = ModelCheckpoint(
    filepath='checkpoint', 
    verbose=1, 
    save_weights_only=False,
    save_freq=20)

history = model.fit(
	image_train, {"age_output": age_train, "gender_output": gender_train, "race_output": race_train},
	validation_data=(image_val, {"age_output": age_val, "gender_output": gender_val, "race_output": race_val}),
	batch_size=32,
	steps_per_epoch=len(image_train) // 32,
	epochs=EPOCHS, verbose=1,
	callbacks=[cp_callback])
```

    Epoch 1/50
     19/244 [=>............................] - ETA: 38:24 - loss: 29.2699 - age_output_loss: 924.0161 - gender_output_loss: 0.5985 - race_output_loss: 0.9509 - age_output_accuracy: 0.0724 - gender_output_accuracy: 0.6711 - race_output_accuracy: 0.6645
    Epoch 00001: saving model to checkpoint
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/resource_variable_ops.py:1817: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    INFO:tensorflow:Assets written to: checkpoint/assets
     39/244 [===>..........................] - ETA: 36:41 - loss: 20.2688 - age_output_loss: 624.6887 - gender_output_loss: 0.5958 - race_output_loss: 0.9323 - age_output_accuracy: 0.0825 - gender_output_accuracy: 0.6851 - race_output_accuracy: 0.6651
    Epoch 00001: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
     59/244 [======>.......................] - ETA: 33:46 - loss: 16.6293 - age_output_loss: 503.2106 - gender_output_loss: 0.5911 - race_output_loss: 0.9419 - age_output_accuracy: 0.0842 - gender_output_accuracy: 0.6827 - race_output_accuracy: 0.6647
    Epoch 00001: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
     79/244 [========>.....................] - ETA: 30:19 - loss: 15.2535 - age_output_loss: 457.4263 - gender_output_loss: 0.5938 - race_output_loss: 0.9369 - age_output_accuracy: 0.0902 - gender_output_accuracy: 0.6780 - race_output_accuracy: 0.6646
    Epoch 00001: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
     99/244 [===========>..................] - ETA: 26:44 - loss: 14.0014 - age_output_loss: 415.8371 - gender_output_loss: 0.5832 - race_output_loss: 0.9431 - age_output_accuracy: 0.0878 - gender_output_accuracy: 0.6891 - race_output_accuracy: 0.6616
    Epoch 00001: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    119/244 [=============>................] - ETA: 23:09 - loss: 13.3412 - age_output_loss: 394.0124 - gender_output_loss: 0.5800 - race_output_loss: 0.9408 - age_output_accuracy: 0.0874 - gender_output_accuracy: 0.6917 - race_output_accuracy: 0.6675
    Epoch 00001: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    139/244 [================>.............] - ETA: 19:28 - loss: 12.8235 - age_output_loss: 376.7086 - gender_output_loss: 0.5769 - race_output_loss: 0.9453 - age_output_accuracy: 0.0850 - gender_output_accuracy: 0.6949 - race_output_accuracy: 0.6659
    Epoch 00001: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    159/244 [==================>...........] - ETA: 15:47 - loss: 12.3316 - age_output_loss: 360.3804 - gender_output_loss: 0.5801 - race_output_loss: 0.9401 - age_output_accuracy: 0.0825 - gender_output_accuracy: 0.6950 - race_output_accuracy: 0.6716
    Epoch 00001: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    179/244 [=====================>........] - ETA: 12:05 - loss: 11.8322 - age_output_loss: 343.4265 - gender_output_loss: 0.5816 - race_output_loss: 0.9478 - age_output_accuracy: 0.0828 - gender_output_accuracy: 0.6940 - race_output_accuracy: 0.6688
    Epoch 00001: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    199/244 [=======================>......] - ETA: 8:22 - loss: 11.5272 - age_output_loss: 333.2910 - gender_output_loss: 0.5798 - race_output_loss: 0.9487 - age_output_accuracy: 0.0804 - gender_output_accuracy: 0.6952 - race_output_accuracy: 0.6698
    Epoch 00001: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    219/244 [=========================>....] - ETA: 4:39 - loss: 11.2538 - age_output_loss: 324.0854 - gender_output_loss: 0.5799 - race_output_loss: 0.9513 - age_output_accuracy: 0.0796 - gender_output_accuracy: 0.6943 - race_output_accuracy: 0.6671
    Epoch 00001: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    239/244 [============================>.] - ETA: 55s - loss: 10.9755 - age_output_loss: 314.9449 - gender_output_loss: 0.5798 - race_output_loss: 0.9474 - age_output_accuracy: 0.0800 - gender_output_accuracy: 0.6944 - race_output_accuracy: 0.6680 
    Epoch 00001: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    244/244 [==============================] - 2927s 12s/step - loss: 10.9471 - age_output_loss: 313.9161 - gender_output_loss: 0.5794 - race_output_loss: 0.9502 - age_output_accuracy: 0.0808 - gender_output_accuracy: 0.6949 - race_output_accuracy: 0.6671 - val_loss: 18.7273 - val_age_output_loss: 572.7776 - val_gender_output_loss: 0.5789 - val_race_output_loss: 0.9651 - val_age_output_accuracy: 0.0169 - val_gender_output_accuracy: 0.6994 - val_race_output_accuracy: 0.6713
    Epoch 2/50
     15/244 [>.............................] - ETA: 39:18 - loss: 10.7096 - age_output_loss: 308.6943 - gender_output_loss: 0.5513 - race_output_loss: 0.8974 - age_output_accuracy: 0.0736 - gender_output_accuracy: 0.7186 - race_output_accuracy: 0.6840
    Epoch 00002: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
     35/244 [===>..........................] - ETA: 37:47 - loss: 8.8685 - age_output_loss: 247.7646 - gender_output_loss: 0.5583 - race_output_loss: 0.8773 - age_output_accuracy: 0.0817 - gender_output_accuracy: 0.7123 - race_output_accuracy: 0.6897
    Epoch 00002: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
     55/244 [=====>........................] - ETA: 34:42 - loss: 8.6804 - age_output_loss: 241.2939 - gender_output_loss: 0.5599 - race_output_loss: 0.8817 - age_output_accuracy: 0.0815 - gender_output_accuracy: 0.7090 - race_output_accuracy: 0.6929
    Epoch 00002: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
     75/244 [========>.....................] - ETA: 31:21 - loss: 8.6301 - age_output_loss: 239.3411 - gender_output_loss: 0.5568 - race_output_loss: 0.8930 - age_output_accuracy: 0.0819 - gender_output_accuracy: 0.7120 - race_output_accuracy: 0.6860
    Epoch 00002: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
     95/244 [==========>...................] - ETA: 27:41 - loss: 8.4367 - age_output_loss: 232.9243 - gender_output_loss: 0.5563 - race_output_loss: 0.8927 - age_output_accuracy: 0.0791 - gender_output_accuracy: 0.7138 - race_output_accuracy: 0.6827
    Epoch 00002: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    115/244 [=============>................] - ETA: 24:01 - loss: 8.5269 - age_output_loss: 236.1605 - gender_output_loss: 0.5597 - race_output_loss: 0.8824 - age_output_accuracy: 0.0800 - gender_output_accuracy: 0.7078 - race_output_accuracy: 0.6903
    Epoch 00002: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    135/244 [===============>..............] - ETA: 20:20 - loss: 8.3889 - age_output_loss: 231.5004 - gender_output_loss: 0.5627 - race_output_loss: 0.8812 - age_output_accuracy: 0.0783 - gender_output_accuracy: 0.7062 - race_output_accuracy: 0.6918
    Epoch 00002: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    155/244 [==================>...........] - ETA: 16:37 - loss: 8.2607 - age_output_loss: 226.8665 - gender_output_loss: 0.5601 - race_output_loss: 0.8946 - age_output_accuracy: 0.0801 - gender_output_accuracy: 0.7094 - race_output_accuracy: 0.6854
    Epoch 00002: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    175/244 [====================>.........] - ETA: 12:54 - loss: 8.1353 - age_output_loss: 222.4826 - gender_output_loss: 0.5622 - race_output_loss: 0.8987 - age_output_accuracy: 0.0808 - gender_output_accuracy: 0.7076 - race_output_accuracy: 0.6842
    Epoch 00002: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    195/244 [======================>.......] - ETA: 9:09 - loss: 7.9410 - age_output_loss: 215.8758 - gender_output_loss: 0.5642 - race_output_loss: 0.9006 - age_output_accuracy: 0.0804 - gender_output_accuracy: 0.7075 - race_output_accuracy: 0.6840
    Epoch 00002: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    215/244 [=========================>....] - ETA: 5:25 - loss: 7.9107 - age_output_loss: 214.8421 - gender_output_loss: 0.5649 - race_output_loss: 0.9005 - age_output_accuracy: 0.0781 - gender_output_accuracy: 0.7056 - race_output_accuracy: 0.6841
    Epoch 00002: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    235/244 [===========================>..] - ETA: 1:41 - loss: 7.8758 - age_output_loss: 213.5597 - gender_output_loss: 0.5670 - race_output_loss: 0.9021 - age_output_accuracy: 0.0784 - gender_output_accuracy: 0.7045 - race_output_accuracy: 0.6830
    Epoch 00002: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    244/244 [==============================] - 2932s 12s/step - loss: 7.8454 - age_output_loss: 212.5322 - gender_output_loss: 0.5680 - race_output_loss: 0.9014 - age_output_accuracy: 0.0773 - gender_output_accuracy: 0.7041 - race_output_accuracy: 0.6843 - val_loss: 20.4274 - val_age_output_loss: 629.6763 - val_gender_output_loss: 0.5821 - val_race_output_loss: 0.9550 - val_age_output_accuracy: 0.1109 - val_gender_output_accuracy: 0.7004 - val_race_output_accuracy: 0.6595
    Epoch 3/50
     11/244 [>.............................] - ETA: 36:11 - loss: 8.1862 - age_output_loss: 219.7931 - gender_output_loss: 0.5764 - race_output_loss: 1.0160 - age_output_accuracy: 0.0868 - gender_output_accuracy: 0.7036 - race_output_accuracy: 0.6467
    Epoch 00003: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
     31/244 [==>...........................] - ETA: 38:02 - loss: 8.5985 - age_output_loss: 236.6638 - gender_output_loss: 0.5731 - race_output_loss: 0.9255 - age_output_accuracy: 0.0903 - gender_output_accuracy: 0.7033 - race_output_accuracy: 0.6828
    Epoch 00003: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
     51/244 [=====>........................] - ETA: 35:16 - loss: 7.8998 - age_output_loss: 215.2543 - gender_output_loss: 0.5534 - race_output_loss: 0.8888 - age_output_accuracy: 0.0892 - gender_output_accuracy: 0.7181 - race_output_accuracy: 0.6902
    Epoch 00003: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
     71/244 [=======>......................] - ETA: 31:53 - loss: 7.8006 - age_output_loss: 212.6448 - gender_output_loss: 0.5569 - race_output_loss: 0.8643 - age_output_accuracy: 0.0874 - gender_output_accuracy: 0.7183 - race_output_accuracy: 0.6988
    Epoch 00003: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
     91/244 [==========>...................] - ETA: 28:23 - loss: 7.7237 - age_output_loss: 209.6027 - gender_output_loss: 0.5575 - race_output_loss: 0.8782 - age_output_accuracy: 0.0867 - gender_output_accuracy: 0.7173 - race_output_accuracy: 0.6938
    Epoch 00003: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    111/244 [============>.................] - ETA: 24:45 - loss: 7.4768 - age_output_loss: 201.6903 - gender_output_loss: 0.5594 - race_output_loss: 0.8666 - age_output_accuracy: 0.0821 - gender_output_accuracy: 0.7170 - race_output_accuracy: 0.6964
    Epoch 00003: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    131/244 [===============>..............] - ETA: 21:06 - loss: 7.3628 - age_output_loss: 197.4697 - gender_output_loss: 0.5652 - race_output_loss: 0.8734 - age_output_accuracy: 0.0839 - gender_output_accuracy: 0.7111 - race_output_accuracy: 0.6929
    Epoch 00003: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    151/244 [=================>............] - ETA: 17:22 - loss: 7.3428 - age_output_loss: 196.8649 - gender_output_loss: 0.5649 - race_output_loss: 0.8720 - age_output_accuracy: 0.0823 - gender_output_accuracy: 0.7084 - race_output_accuracy: 0.6926
    Epoch 00003: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    171/244 [====================>.........] - ETA: 13:39 - loss: 7.4604 - age_output_loss: 200.8791 - gender_output_loss: 0.5615 - race_output_loss: 0.8726 - age_output_accuracy: 0.0812 - gender_output_accuracy: 0.7105 - race_output_accuracy: 0.6929
    Epoch 00003: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    191/244 [======================>.......] - ETA: 9:55 - loss: 7.4045 - age_output_loss: 199.3676 - gender_output_loss: 0.5589 - race_output_loss: 0.8645 - age_output_accuracy: 0.0801 - gender_output_accuracy: 0.7137 - race_output_accuracy: 0.6945 
    Epoch 00003: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    211/244 [========================>.....] - ETA: 6:10 - loss: 7.5255 - age_output_loss: 203.4311 - gender_output_loss: 0.5561 - race_output_loss: 0.8665 - age_output_accuracy: 0.0781 - gender_output_accuracy: 0.7155 - race_output_accuracy: 0.6950
    Epoch 00003: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    231/244 [===========================>..] - ETA: 2:26 - loss: 7.5564 - age_output_loss: 204.2778 - gender_output_loss: 0.5559 - race_output_loss: 0.8722 - age_output_accuracy: 0.0785 - gender_output_accuracy: 0.7160 - race_output_accuracy: 0.6935
    Epoch 00003: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
    244/244 [==============================] - 2933s 12s/step - loss: 7.5258 - age_output_loss: 203.2711 - gender_output_loss: 0.5548 - race_output_loss: 0.8728 - age_output_accuracy: 0.0783 - gender_output_accuracy: 0.7158 - race_output_accuracy: 0.6933 - val_loss: 7.2711 - val_age_output_loss: 192.4794 - val_gender_output_loss: 0.5810 - val_race_output_loss: 0.9158 - val_age_output_accuracy: 0.0409 - val_gender_output_accuracy: 0.7035 - val_race_output_accuracy: 0.6820
    Epoch 4/50
      7/244 [..............................] - ETA: 33:09 - loss: 8.0997 - age_output_loss: 226.5923 - gender_output_loss: 0.5286 - race_output_loss: 0.7734 - age_output_accuracy: 0.0583 - gender_output_accuracy: 0.7524 - race_output_accuracy: 0.7524
    Epoch 00004: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
     27/244 [==>...........................] - ETA: 38:00 - loss: 7.2978 - age_output_loss: 197.9445 - gender_output_loss: 0.5325 - race_output_loss: 0.8270 - age_output_accuracy: 0.0816 - gender_output_accuracy: 0.7376 - race_output_accuracy: 0.7222
    Epoch 00004: saving model to checkpoint
    INFO:tensorflow:Assets written to: checkpoint/assets
     47/244 [====>.........................] - ETA: 35:38 - loss: 7.1844 - age_output_loss: 193.5383 - gender_output_loss: 0.5402 - race_output_loss: 0.8380 - age_output_accuracy: 0.0794 - gender_output_accuracy: 0.7436 - race_output_accuracy: 0.7160
    Epoch 00004: saving model to checkpoint
    

### 4.3.4. Huấn luyện lại mô hình

Quá trình huấn luyện sẽ khá tốn thời gian. Mình huấn luyện 4 epochs trên google colab hết khoảng 2 tiếng. Để huấn luyện lại mô hình từ checkpoint các bạn chỉ cần load lại model. Các trạng thái của optimizer như learning rate, gradient descent đã được lưu lại. Chúng ta sẽ chỉ cần retrain lại mô hình.


```
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

LR_RATE = 0.001
EPOCHS = 50
opt = Adam(lr=LR_RATE, decay=LR_RATE / EPOCHS)
model = load_model('checkpoint')
model.compile(loss=losses, loss_weights= lossWeights, optimizer=opt, metrics=['accuracy'])

cp_callback = ModelCheckpoint(
    filepath='checkpoint', 
    verbose=1, 
    save_weights_only=False,
    save_freq=20)

history = model.fit(
	image_train, {"age_output": age_train, "gender_output": gender_train, "race_output": race_train},
	validation_data=(image_val, {"age_output": age_val, "gender_output": gender_val, "race_output": race_val}),
	batch_size=32,
	steps_per_epoch=len(image_train) // 32,
	epochs=EPOCHS, verbose=1,
	callbacks=[cp_callback])
```

### 4.3.5. Dự báo độ tuổi, giới tính và chủng tộc

Từ mô hình sau huẩn luyện, chúng ta có thể load lại chúng từ checkpoint và dự báo cho đồng thời các tác vụ. Output của chúng ta sẽ là một tupple gồm 3 outputs tương ứng với từng tác vụ dự báo `age, gender, race`. Đối với age, kết quả sẽ được làm tròn phần nguyên để ước lượng độ tuổi. Đối với gender và race, nhãn của output sẽ là phần tử có xác suất lớn nhất. Phần này xin dành cho bạn đọc như một bài tập.

# 5. Hướng mở rộng

Như vậy mô hình MultiTaskLearning sử dụng nhiều branch đã có khả năng dự báo được nhiều đặc tính trên khuôn mặt đồng thời và đạt kết quả khá tốt. Thực tế cho thấy trong 3 tác vụ dự báo thì age có khả năng nhầm lẫn cao nhất vì age chứa nhiều ảnh nhiễu. Có người có thể trông già trước tuổi hoặc trẻ hơn tuổi. Đồng thời số lượng classes của age cũng là lớn nhất. 

Do đó để cải thiện độ chính xác cho age, bạn đọc có thể thử nghiệm thêm nhiều hàm loss function khác bằng cách coi dự báo age làm một bài toán phân loại. Khi đó output sẽ là một véc tơ phân phối xác suất $\hat{p}_{i} = (\hat{p}_{i1}, ... ,\hat{p}_{iK})$ với $p_{ij}$ tương ứng với xác suất rơi vào tuổi $j$ của người $i$. Bên dưới là một vài hàm loss function mà mình nghĩ ra:

* **Hàm MSE**: Nếu coi giá trị dự báo của bài toán classification là trung bình có trọng số $\bar{y}_{i} = \sum_{j=1}^{K}(j.\hat{p}_{ij})$ theo phân phối xác suất thì hàm loss function dạng MSE sẽ là:

$$\mathcal{L}(\mathbf{X}; \mathbf{W}) = \frac{1}{N}\sum_{i=1}^{N}(\bar{y}_{i}-y_{i})^2$$

* **Hàm Variance**: Nếu coi tuổi thực $y_i$ được phân rã về từng nhóm tuổi $j$ với xác suất là $\hat{p}_{ij}$. Khi đó phương sai giữa giá trị sau phân rã và tuổi thực sẽ là:
$$\mathcal{L}(\mathbf{X}; \mathbf{W}) = \sum_{i=1}^{N}\sum_{j=1}^{K}\hat{p}_{ij}(j-\hat{p}_{ij}*y_{i})^2$$


* **Hàm MSE cross entropy**: Tương tự như hàm variance nhưng ta lấy logarith của độ tuổi. Lấy cảm hứng từ sự kết hợp giữa hàm MSE và cross entropy.

$$\mathcal{L}(\mathbf{X}; \mathbf{W}) = \sum_{i=1}^{N}\sum_{j=1}^{K}\hat{p}_{ij}(\log{(j)}-\hat{p}_{ij}*\log(y_{i}))^2$$


Việc huấn luyện với các hàm số trên xin dành cho bạn đọc như một bài toán mở. Nếu kết quả tốt, bạn có thể viết báo nhé. Đừng quên cảm ơn mình vì đã gợi ý cho bạn.


Ngoài ra bạn đọc cũng có thể dựa trên ý tưởng của thuật toán được trình bày tại bài viết này để thực hành lại [Bài 34 - Multitask Learning](https://phamdinhkhanh.github.io/2020/04/22/MultitaskLearning.html) và so sánh hiệu quả giữa 2 phương pháp rẽ nhánh và chia sẻ tham số.

# 6. Tổng kết

Như vậy sau bài viết này mình đã giới thiệu tới các bạn một dạng tiếp cận mới cho bài toán multitask learning trong trường hợp các tác vụ huấn luyện không chia sẻ đặc trưng phân loại. Kiến trúc multi-branch sẽ hiệu quả hơn so với kiến trúc multi-label output ở bài trước.

Đồng thời bạn đọc cũng được tiếp cận với một số định hướng mở rộng bài toán bằng việc thay đổi các dạng hàm loss function khác nhau để cải thiện độ chính xác của mô hình dự báo.

# 7. Tài liệu

1. [Keras: Multiple outputs and multiple losses - Pyimagesearch](https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/)

2. [Deep Convolutional Neural Network for Age Estimation based on
VGG-Face Model](https://arxiv.org/ftp/arxiv/papers/1709/1709.01664.pdf)

3. [Age estimation - paper with code](https://paperswithcode.com/task/age-estimation/codeless?page=3)

4. [Age estimation via face images: a survey - Springer](https://link.springer.com/article/10.1186/s13640-018-0278-6)

5. [FairFace: Face Attribute Dataset for Balanced Race, Gender, Age](https://arxiv.org/pdf/1908.04913.pdf)
