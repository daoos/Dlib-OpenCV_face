import dlib
import cv2
import numpy as np


def dlib_get_face(frame_path, face_rec_model_path, predictor_path):
    # 使用dlib自带的frontal_face_detector作为人脸提取器
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(predictor_path)
    face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

    # opencv读取图片，并显示
    frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)

    # opencv的bgr格式图片转换成rgb格式
    b, g, r = cv2.split(frame)
    frame2 = cv2.merge([r, g, b])

    # 使用detector进行人脸检测dets为返回的结果
    dets = detector(frame, 1)

    face_descriptor = []
    for index, face in enumerate(dets):
        shape = shape_predictor(frame, face)  # 提取68个特征点
        for i in range(68):
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 5, (0, 255, 0), -1, 8)
            # cv2.putText(frame, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (255, 255, 255))
        # 计算人脸的128维的向量
        face_descriptor.append(face_rec_model.compute_face_descriptor(frame2, shape))
    cv2.imwrite('savelogo.jpg', frame)
    return np.array(face_descriptor)


frame_path = "2070530151.jpg"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
predictor_path = "shape_predictor_68_face_landmarks.dat"
face = dlib_get_face(frame_path, face_rec_model_path, predictor_path)
print(face.shape)