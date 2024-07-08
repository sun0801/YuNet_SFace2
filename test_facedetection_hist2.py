import copy
import math
import argparse
import os
import pdb
import cv2 as cv
import numpy as np
import glob
from utils import CvFpsCalc
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    args = parser.parse_args()
    return args

COSINE_THRESHOLD = 0.363
NORML2_THRESHOLD = 1.128

def match(recognizer, feature1, dictionary, directory):
    max_score = 0.0
    best_match_user_id = ""
    scores = []

    for element in dictionary:
        user_id, feature2 = element
        score = recognizer.match(feature1, feature2, cv.FaceRecognizerSF_FR_COSINE)
        if score > COSINE_THRESHOLD:
            scores.append((user_id, score))
            if score > max_score:
                max_score = score
                best_match_user_id = user_id
    
    if best_match_user_id:
        return True, (best_match_user_id, max_score), scores
    
    # マッチしなかった場合、新しいIDを生成して辞書に追加
    new_user_id = f"user_{len(dictionary) + 1}"
    dictionary.append((new_user_id, feature1))
    save_feature(new_user_id, feature1, directory)
    scores.append((new_user_id, 0.0))
    return False, (new_user_id, 0.0), scores

def save_feature(user_id, feature, directory):
    filename = os.path.join(directory, f"{user_id}.npy")
    np.save(filename, feature)

def plot_histogram(scores):
    plt.clf()
    user_ids, match_scores = zip(*scores)
    plt.figure(figsize=(10, 5))
    plt.bar(user_ids, match_scores, color='blue')
    plt.xlabel('User ID')
    plt.ylabel('Match Score')
    plt.title('Match Scores for Detected Faces')
    plt.ylim(0, 1)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.pause(.01)
    plt.close()

def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    directory = os.path.dirname(__file__)
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    dictionary = []
    files = glob.glob(os.path.join(directory, "*.npy"))
    for file in files:
        feature = np.load(file)
        user_id = os.path.splitext(os.path.basename(file))[0]
        dictionary.append((user_id, feature))

    weights = os.path.join(directory, "face_detection_yunet_2023mar.onnx")
    face_detection = cv.FaceDetectorYN_create(weights, "", (0, 0))
    weights = os.path.join(directory, "face_recognizer_fast.onnx")
    face_recognizer = cv.FaceRecognizerSF_create(weights, "")

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        display_fps = cvFpsCalc.get()

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)

        height, width, _ = image.shape
        face_detection.setInputSize((width, height))

        result, faces = face_detection.detect(image)
        faces = faces if faces is not None else []

        for face in faces:
            aligned_face = face_recognizer.alignCrop(image, face)
            feature = face_recognizer.feature(aligned_face)

            matched, user, scores = match(face_recognizer, feature, dictionary, directory)

            box = list(map(int, face[:4]))
            color = (0, 255, 0) if matched else (0, 0, 255)
            thickness = 2
            cv.rectangle(image, box, color, thickness, cv.LINE_AA)

            user_id, score = user
            text = "{0} ({1:.2f})".format(user_id, score)
            position = (box[0], box[1] - 10)
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            cv.putText(image, text, position, font, scale, color, thickness, cv.LINE_AA)

            # ヒストグラムのプロットを更新
            plot_histogram(scores)

        cv.imshow("face recognition", image)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    plt.close()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()