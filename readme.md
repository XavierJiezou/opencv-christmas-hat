# 成果展示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201230003223916.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTUxNTYw,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201230003221839.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTUxNTYw,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201230003234780.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTUxNTYw,size_16,color_FFFFFF,t_70)

# Introduce
The principle is simple : First, detect faces in the image, and then add chritmas hat above the faces.

# Preparation
> [【python】15 lines of code to achieve human-face detection（opencv）](https://blog.csdn.net/qq_42951560/article/details/111694348)

![](https://img-blog.csdnimg.cn/20201228094823894.jpg)
> [【python】15 lines of code to achieve cat-face detection（opencv）](https://blog.csdn.net/qq_42951560/article/details/111831532)

![](https://img-blog.csdnimg.cn/20201228102022683.jpg)
> [【python】15 lines of code to achieve anime-face detection（opencv）](https://blog.csdn.net/qq_42951560/article/details/111831797)

![](https://img-blog.csdnimg.cn/20201228103025477.jpg)

# Codes
```python
import cv2
import random


def face_detect(img, cname):
    # 脸部检测代码
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hist = cv2.equalizeHist(img_gray)
    face_cascade = cv2.CascadeClassifier(cname)
    faces = face_cascade.detectMultiScale(img_hist)
    return faces


def christmas_hat(fname, cname):
    img = cv2.imread(fname)
    # 脸部检测
    faces = face_detect(img, cname)
    # 读取圣诞帽
    hats = [cv2.imread(f'img/hats/hat_{i+1}.png', -1) for i in range(3)]
    for face in faces:
        hat = random.choice(hats) # 随机选择一个帽子
        scale = face[3] / hat.shape[0] * 2  # 设置缩放因子
        hat = cv2.resize(hat, (0, 0), fx=scale, fy=scale) # 调整帽子大小
        x_offset = int(face[0] + face[2] / 2 - hat.shape[1] / 2) # 计算帽子的x偏移
        y_offset = int(face[1] - hat.shape[0] / 2)  # 计算帽子的y偏移
        # 计算贴图位置，注意防止超出边界的情况
        x1 = max(x_offset, 0)
        x2 = min(x_offset + hat.shape[1], img.shape[1])
        y1 = max(y_offset, 0)
        y2 = min(y_offset + hat.shape[0], img.shape[0])
        hat_x1 = max(0, -x_offset)
        hat_x2 = hat_x1 + x2 - x1
        hat_y1 = max(0, -y_offset)
        hat_y2 = hat_y1 + y2 - y1
        # 透明部分的处理
        alpha_h = hat[hat_y1:hat_y2, hat_x1:hat_x2, 3] / 255
        alpha = 1 - alpha_h
        # 按3个通道合并图片
        for c in range(3):
            img[y1:y2, x1:x2, c] = alpha_h * hat[hat_y1:hat_y2, hat_x1:hat_x2, c] + alpha * img[y1:y2, x1:x2, c]
        # 保存最终结果
        cv2.imwrite(f'img/result/{fname.split("/")[-1]}', img) 


def main(fname):
    target = input('请选择要添加圣诞帽的对象: 1 人 2 猫 3 动漫 (默认1) ')
    target = target if target else '1'
    if target == '1':
        cname = 'data/haarcascade_frontalface_alt.xml'
    elif target == '2':
        cname = 'data/haarcascade_frontalcatface.xml'
    elif target == '3':
        cname = 'data/lbpcascade_animeface.xml'
    else:
        print('检测对象输入有误')
    christmas_hat(fname, cname)


if __name__ == "__main__":
    main('img/test/test_3.jpg') # 这里输入要添加圣诞帽的图像路径
```
# Requirements
The following three files are the cascade classifiers of human face, cat face and anime face:
- `haarcascade_frontalface_alt.xml`：[download](https://cdn.jsdelivr.net/gh/XavierJiezou/opecv-face-detect@master/data/haarcascades/human/haarcascade_frontalface_alt.xml)
- `haarcascade_frontalcatface.xml`：[download](https://cdn.jsdelivr.net/gh/XavierJiezou/opecv-face-detect@master/data/haarcascades/cat/haarcascade_frontalcatface.xml)
- `lbpcascade_animeface.xml`：[download](https://cdn.jsdelivr.net/gh/XavierJiezou/opecv-face-detect@master/data/lbpcascades/anime/lbpcascade_animeface.xml)
# Christmas material
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020123000442237.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTUxNTYw,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201230004422101.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTUxNTYw,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201230004423774.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyOTUxNTYw,size_16,color_FFFFFF,t_70)
# Cite
> https://github.com/crossin/snippet/tree/master/face_detect

# Recommend
> [【python】30 lines of code to achieve anime-face detection in the video（opencv）](https://blog.csdn.net/qq_42951560/article/details/111870163)

![](https://img-blog.csdnimg.cn/20201228165341951.gif)