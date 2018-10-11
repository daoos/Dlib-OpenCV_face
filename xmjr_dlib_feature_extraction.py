import dlib
import cv2
import numpy as np
import cv2
import dlib


# def dlib_get_face(frame_path, face_rec_model_path, predictor_path):
#     # 使用dlib自带的frontal_face_detector作为人脸提取器
#     detector = dlib.get_frontal_face_detector()
#     shape_predictor = dlib.shape_predictor(predictor_path)
#     face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
#
#     # opencv读取图片，并显示
#     frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
#
#     # opencv的bgr格式图片转换成rgb格式
#     b, g, r = cv2.split(frame)
#     frame2 = cv2.merge([r, g, b])
#
#     # 使用detector进行人脸检测dets为返回的结果
#     dets = detector(frame, 1)
#
#     face_descriptor = []
#     for index, face in enumerate(dets):
#         shape = shape_predictor(frame, face)  # 提取68个特征点
#         # 计算人脸的128维的向量
#         face_descriptor.append(face_rec_model.compute_face_descriptor(frame2, shape))
#     return np.array(face_descriptor)


def dlib_get_face(frame, face_rec_model_path, predictor_path):
    # 使用dlib自带的frontal_face_detector作为人脸提取器
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(predictor_path)
    face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

    # opencv读取图片，并显示

    # opencv的bgr格式图片转换成rgb格式
    b, g, r = cv2.split(frame)
    frame2 = cv2.merge([r, g, b])

    # 使用detector进行人脸检测dets为返回的结果
    dets = detector(frame, 1)

    face_descriptor = []
    for index, face in enumerate(dets):
        shape = shape_predictor(frame, face)  # 提取68个特征点
        # 计算人脸的128维的向量
        face_descriptor.append(face_rec_model.compute_face_descriptor(frame2, shape))
    return np.array(face_descriptor)



# frame_path = "a.jpeg"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
predictor_path = "shape_predictor_68_face_landmarks.dat"



PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# 使用dlib自带的frontal_face_detector作为人脸提取器
detector = dlib.get_frontal_face_detector()

# 使用官方模型构建特征提取器
predictor = dlib.shape_predictor(PREDICTOR_PATH)

if __name__ == '__main__':

    #框住人脸的矩形边框颜色
    color = (0, 255, 0)

    #捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)

    #循环检测识别人脸
    while True:
        _, frame = cap.read()   #读取一帧视频

        # 图像灰化，降低计算复杂度
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用detector进行人脸检测 rects为返回的结果
        rects = detector(frame_gray, 1)

        if len(rects) > 0:
            for k, d in enumerate(rects):
                cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255))
                shape = predictor(frame, d)
                for i in range(68):
                    cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 5, (0, 255, 0), -1, 8)
                    # cv2.putText(frame, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    #             (255, 255, 255))
        face = dlib_get_face(frame, face_rec_model_path, predictor_path)
        print(face)
        cv2.imshow("find me", frame)

        #等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        #如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
