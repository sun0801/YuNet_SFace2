import copy
import math
import argparse
import os
import pdb

import cv2 as cv
import numpy as np
import glob

from utils import CvFpsCalc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    args = parser.parse_args()
    return args

COSINE_THRESHOLD = 0.363
NORML2_THRESHOLD = 1.128

directory = os.path.dirname(__file__)

# 特徴を辞書と比較してマッチしたユーザーとスコアを返す関数
def match(recognizer, feature1, dictionary):
    for element in dictionary:
        user_id, feature2 = element
        score = recognizer.match(feature1, feature2, cv.FaceRecognizerSF_FR_COSINE)
        if score > COSINE_THRESHOLD:
            return True, (user_id, score)
        
    # マッチしなかった場合、新しいIDを生成して辞書に追加
    new_user_id = f"user_{len(dictionary) + 1}"
    dictionary.append((new_user_id, feature1))
    save_feature(new_user_id, feature1, directory)
    return False, (new_user_id, 0.0)
    return False, ("", 0.0)

def save_feature(user_id, feature, directory):
    filename = os.path.join(directory, f"{user_id}.npy")
    np.save(filename, feature)

def main():
    # 引数解析 #################################################################
    # pdb.set_trace()
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    # cap = cv.VideoCapture()
    # cap.open('http://10.40.0.144:4747/video')  # OK (480,640,3)

    # 特徴を読み込む
    dictionary = []
    files = glob.glob(os.path.join(directory, "*.npy"))
    for file in files:
        feature = np.load(file)
        user_id = os.path.splitext(os.path.basename(file))[0]
        dictionary.append((user_id, feature))

    # モデルロード #############################################################
    weights = os.path.join(directory, "face_detection_yunet_2023mar.onnx")
    face_detection = cv.FaceDetectorYN_create(weights, "", (0, 0))
    weights = os.path.join(directory, "face_recognizer_fast.onnx")
    face_recognizer = cv.FaceRecognizerSF_create(weights, "")

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 画像が3チャンネル以外の場合は3チャンネルに変換する
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)

        # 入力サイズを指定する
        height, width, _ = image.shape
        face_detection.setInputSize((width, height))

        # 検出実施 #############################################################
        result, faces = face_detection.detect(image)
        faces = faces if faces is not None else []

        # 顔識別 ###############################################################
        for face in faces:
            # 顔を切り抜き特徴を抽出する
            aligned_face = face_recognizer.alignCrop(image, face)
            feature = face_recognizer.feature(aligned_face)

            # 辞書とマッチングする
            matched, user = match(face_recognizer, feature, dictionary)

            # 顔のバウンディングボックスを描画する
            box = list(map(int, face[:4]))
            color = (0, 255, 0) if result else (0, 0, 255)
            thickness = 2
            cv.rectangle(image, box, color, thickness, cv.LINE_AA)

            # 認識の結果を描画する
            user_id, score = user
            text = "{0} ({1:.2f})".format(user_id, score)
            position = (box[0], box[1] - 10)
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            cv.putText(image, text, position, font, scale, color, thickness, cv.LINE_AA)

        # 画像を表示する
        cv.imshow("face recognition", image)
        key = cv.waitKey(1)
        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()