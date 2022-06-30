import numpy as np
import cv2
import dlib
import keras

detector = dlib.get_frontal_face_detector() #构建人脸框位置检测器

class Recognition:
    def __init__(self):
        self.model = None
        self.face_id_list = []
        self.my_name = ''
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def face_recognition(self, model_path, stream):
        self.load_model(model_path)
        while stream.isOpened():
            ret, img_camera = stream.read()
            img_gray = cv2.cvtColor(img_camera, cv2.COLOR_BGR2GRAY)
            faces = detector(img_gray, 1)
            if len(faces) != 0:
                for i, face in enumerate(faces):
                    left = face.left()
                    right = face.right()
                    top = face.top()
                    bottom = face.bottom()

                    img_cut = img_camera[top: bottom, left: right]
                    if img_cut.shape[0] != 0 and img_cut.shape[1] != 0:
                        after_resize = cv2.resize(img_cut, (64, 64))
                        cv2.rectangle(img_camera, (left, top), (right, bottom), (0, 255, 0), 2)
                        faceID = self.face_predict(after_resize)  # 得到此帧人脸对应ID
                        if len(self.face_id_list) < 20:
                            self.face_id_list.append(faceID)  # 将预测的人脸ID加入到列表当中
                        else:
                            print(self.face_id_list)
                            max_label = max(self.face_id_list, key=self.face_id_list.count)
                            img_set_id = "{}".format(max_label)
                            self.my_name = str(img_set_id)
                            self.face_id_list = []
            cv2.putText(img_camera, self.my_name, (left, top - 10), self.font, 1, (0, 255, 0), 1)
            cv2.imshow("camera", img_camera)
            if (cv2.waitKey(1) == ord('q')):
                break

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)

    def face_predict(self, image):
        image = np.array(image)
        image = image.reshape((1, 64, 64, 3))
        image = image.astype('float32')
        image /= 255
        result = self.model.predict_classes(image)
        return result[0]

    def run(self):
        model_path = '../data/model/model.h5'
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.face_recognition(model_path, cap)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    predict = Recognition()
    predict.run()
