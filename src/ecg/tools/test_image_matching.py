from imcui.api.core import ImageMatchingAPI
import cv2
from imcui.ui.viz import display_matches



# ref: https://github.com/Vincentqyw/image-matching-webui/blob/main/imcui/hloc/configs/matchers.py
conf = {
    "ransac": {
        "enable": True,
        "estimator": "poselib",
        "geometry": "homography",
        "method": "cv2_USAC_MAGSAC",
        "reproj_threshold": 2,
        "confidence": 0.9999,
        "max_iter": 100000,
    },
    "matcher": {
        "output": "matches-minima_roma",
        "model": {
            "name": "roma",
            "weights": "outdoor",
            "model_name": "minima_roma.pth",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
        },
        "preprocessing": {
            "grayscale": False,
            "force_resize": True,
            "resize_max": 1536,
            "width": 320,
            "height": 240,
            "dfactor": 8,
        },
    },
    "dense": True,
}

api = ImageMatchingAPI(conf=conf, device="cuda:0")

print(api)
print(api.conf)
# exit()


img_path1 = "/home/dangnh36/datasets/ecg/raw/train/514629128/514629128-0001.png"
img_path2 = "/home/dangnh36/datasets/ecg/raw/train/514629128/514629128-0005.png"
image0 = cv2.imread(str(img_path1))[:, :, ::-1]  # RGB
image1 = cv2.imread(str(img_path2))[:, :, ::-1]  # RGB

pred = api(image0, image1)
assert pred is not None
# print('PRED:', pred, sep='\n')
print(pred.keys())
titles = ["Image 0 - RANSAC matched keypoints", "Image 1 - RANSAC matched keypoints"]
output_matches_ransac, _ = display_matches(pred, titles=titles, tag="KPTS_RANSAC")
cv2.imwrite("/home/dangnh36/downloads/demo/match.png", output_matches_ransac)
