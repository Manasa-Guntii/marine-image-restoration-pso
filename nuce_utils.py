import numpy as np
import cv2 as cv
from PSO import pso
from skimage.metrics import structural_similarity as ssim


def superior_inferior_split(img):

    B, G, R = cv.split(img)

    pixel = {"B": np.mean(B), "G": np.mean(G), "R": np.mean(R)}

    pixel_ordered = sorted(pixel.items(), key=lambda x: x[1], reverse=True)

    channels = {"Pmax": None, "Pint": None, "Pmin": None}
    labels = ["Pmax", "Pint", "Pmin"]

    for i, (name, _) in enumerate(pixel_ordered):

        if name == "B":
            channels[labels[i]] = [B, "B"]

        elif name == "G":
            channels[labels[i]] = [G, "G"]

        else:
            channels[labels[i]] = [R, "R"]

    return channels


def stack_image(channels):

    for val in channels:

        if val[1] == "B":
            b = val[0]

        elif val[1] == "G":
            g = val[0]

        else:
            r = val[0]

    return np.dstack([b, g, r]).astype(np.uint8)


def neutralize_image(img):

    track = superior_inferior_split(img)

    Pmax = track["Pmax"][0]
    Pint = track["Pint"][0]
    Pmin = track["Pmin"][0]

    J = (np.sum(Pmax) - np.sum(Pint)) / (np.sum(Pmax) + np.sum(Pint) + 1e-6)
    K = (np.sum(Pmax) - np.sum(Pmin)) / (np.sum(Pmax) + np.sum(Pmin) + 1e-6)

    track["Pint"][0] = np.clip(Pint + J * Pmax, 0, 255)
    track["Pmin"][0] = np.clip(Pmin + K * Pmax, 0, 255)

    return stack_image(track.values())


def Stretching(image):

    result = []

    for c in range(3):

        channel = image[:, :, c]

        stretched = cv.normalize(channel, None, 0, 255, cv.NORM_MINMAX)

        result.append(stretched)

    img = np.dstack(result).astype(np.uint8)

    return img, img


def enhanced_image(img1, img2):

    return cv.addWeighted(img1, 0.5, img2, 0.5, 0)


def pso_image(img):

    group = superior_inferior_split(img)

    maxi = np.mean(group["Pmax"][0])
    inte = np.mean(group["Pint"][0])
    mini = np.mean(group["Pmin"][0])

    params = {"wmax": 0.9, "wmin": 0.4, "c1": 2, "c2": 2}

    def fitness(X):
        return (maxi - X[0]) ** 2 + (maxi - X[1]) ** 2

    gbest = pso(fitness, 50, 30, 2, 0, 255, params)

    mean_colors = gbest["position"]

    gamma = np.log((mean_colors + 1) / 255) / np.log((np.array([inte, mini]) + 1) / 255)

    gamma = np.clip(gamma, 0.6, 1.4)

    group["Pint"][0] = 255 * np.power(group["Pint"][0] / 255, gamma[0])
    group["Pmin"][0] = 255 * np.power(group["Pmin"][0] / 255, gamma[1])

    return stack_image(group.values())


def unsharp_masking(img, alpha=0.2):

    blur = cv.GaussianBlur(img, (5, 5), 1)

    return cv.addWeighted(img, 1.2, blur, -0.2, 0)


def NUCE(img):

    neu = neutralize_image(img)

    img1, img2 = Stretching(neu)

    dual = enhanced_image(img1, img2)

    pso_res = pso_image(dual)

    final = unsharp_masking(pso_res)

    return final


def calculate_psnr(img1, img2):

    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

    if mse == 0:
        return 100

    PIXEL_MAX = 255.0

    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def calculate_ssim(img1, img2):

    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    score, _ = ssim(gray1, gray2, full=True)

    return score