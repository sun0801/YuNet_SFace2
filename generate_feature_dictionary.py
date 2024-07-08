import os
import sys
import argparse
import numpy as np
import cv2

def main():
    # 引数をパースする
    parser = argparse.ArgumentParser("generate face feature dictionary from an face image")
    parser.add_argument("image", help="input face image file path (./face.jpg)")
    args = parser.parse_args()

    # 引数から画像ファイルのパスを取得
    path = args.image
    directory = os.path.dirname(args.image)
    if not directory:
        directory = os.path.dirname(__file__)
        path = os.path.join(directory, args.image)

    # 画像を開く
    image = cv2.imread(path)
    if image is None:
        exit()

    # 画像が3チャンネル以外の場合は3チャンネルに変換する
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # モデルを読み込む
    weights = os.path.join(directory, "face_recognizer_fast.onnx")
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    # 特徴を抽出する
    face_feature = face_recognizer.feature(image)
    print("face_feature : " + str(face_feature))
    print(type(face_feature))

    # 特徴を保存する
    basename = os.path.splitext(os.path.basename(args.image))[0]
    print("basename : " + str(basename))
    dictionary = os.path.join(directory, basename)
    print("dictionary : " + str(dictionary))
    np.save(dictionary, face_feature)

if __name__ == '__main__':
    main()