# pyright: reportUnboundVariable=false
# pyright: reportUndefinedVariable=false

import numpy as np
import pickle
from skimage.measure import profile_line
from scipy.signal import find_peaks
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from labtool_ex2 import Project


# (0,0)-----------(x,0)
# |-------------------|
# |-------------------|
# |-------------------|
# |-------------------|
# |-------------------|
# (0,y)-----------(x,y)
# reference pictures Hg_01 and Hg_02: 5184 x 3456
# pixel position at the right end of the red line with closest HEX code possible
# y val of pixel 1350
# 558 left val right val 4710
pic_width = 5184
pic_height = 3456
# Using a picture editing tool find non overlapping span using any of
# the Mercury spectral lines which appears in two pictures
loc1 = 4610
loc2 = 560
cropped_picture_span = loc1 - loc2
links_lastig = 1  # a value between 0 and 1
x_start = (pic_width - cropped_picture_span) * (1 - links_lastig)
x_end = (pic_width - cropped_picture_span) * (1 - links_lastig) + cropped_picture_span
y_start = 1350
y_end = 2500
panowidth = 8 * cropped_picture_span
panoheight = y_end - y_start

DATAPATH = "./data/"
PICTURE_DIR = f"{DATAPATH}edited/"


def getIntensity(greyscale_img, profile_line_height) -> np.ndarray:
    intensity = profile_line(
        greyscale_img,
        src=(profile_line_height, 0),
        dst=(profile_line_height, panowidth),
        linewidth=50,
    )
    return intensity


# (x_start,y_start)----D------(x_end,y_start) <<<
# |--------------------D--------------------| <<<
# |--------------------D--------------------| <<<
# |-----------------------------------------|
# |-----------------------------------------|
# |-----------------------------------------|
# (x_start,y_end)---------------(x_end,y_end)
scanline_height = panoheight // 7


def analysis(P: Project, peaks: NDArray, pan: NDArray, name: str):
    intensity = getIntensity(peaks, scanline_height)
    print(intensity.shape)

    # pickle intensities
    with open(f"{DATAPATH}/{name}pickle", "wb") as file:
        pickle.dump(intensity, file)

    # plot the whole thing
    # fig, ax = plt.subplots(nrows=2, sharex=True)
    pxToNm, nmToPx = genBasisMap(P)

    P.figure.clear()
    ax_picture: plt.Axes = P.figure.add_subplot(2, 1, 1)
    ax_int: plt.Axes = P.figure.add_subplot(2, 1, 2)

    P.vload()

    extra_ax = ax_int.twiny()

    lticks = np.arange(int(pxToNm(0)) + 1, int(pxToNm(len(intensity))), 20)
    pxTicks = nmToPx(lticks)
    lxTicks = pxToNm(pxTicks)

    ax_picture.imshow(pan, aspect="auto")
    ax_picture.set_ylabel("$p$ / px")
    ax_picture.set_xlim(0, len(intensity))
    P.data = pd.DataFrame(None)

    P.data["I"] = intensity
    P.data["px"] = np.arange(len(intensity))

    P.plot(
        ax_int,
        px,
        I,
        label="Intensität",
        style="#1cb2f5",
    )

    ax_int.set_xlim(0, len(intensity))
    extra_ax.set_xlim(ax_int.get_xlim())
    extra_ax.set_xticks(pxTicks)
    extra_ax.set_xticklabels(map(str, lticks))
    extra_ax.set_xlabel(r"$\lambda$ / \si{\nano\meter}")
    # ax[1].grid()
    P.figure.tight_layout(pad=1)
    P.figure.subplots_adjust(top=0.90)

    P.savefig(f"{name}plot.pdf")

    print(f"{name[:-1]} done")


# KNOWN_SPECTRAL_LINES = [467.8, 479.9, 508.5, 546, 576.9, 579, 643.9]
# # nm Blue, Cyan, Turquoise, Green, Yellow1, Yellow2, Red
KNOWN_SPECTRAL_LINES = [467.8, 508.5, 546, 576.9, 579, 643.9]
# nm Blue, Cyan, Turquoise, Green, Yellow1, Yellow2, Red
OUTPUTPATH = "./output/"


def genBasisMap(P: Project) -> tuple[callable, callable]:
    P.figure.clear()
    P.data = pd.DataFrame(None)
    ax = P.figure.add_subplot()
    with open("./data/Hg_pickle", "rb") as file:
        intensity = pickle.load(file)

    distance = 130  # Minimum distance before another peak is searched
    height = 11  # height of lowest peak
    peaks, *_ = find_peaks(intensity, distance=distance, height=height)
    P.vload()
    violet, violet2, *peaks = peaks
    P.data["px"] = peaks
    P.data["l"] = KNOWN_SPECTRAL_LINES
    print(P.data)
    P.plot_data(
        axes=ax,
        x=px,
        y=l,
        label="Referenzwerte",
        style="#da93ea",
    )
    l = a * (px - b) ** 2 + c

    params = P.plot_fit(
        axes=ax,
        x=px,
        y=l,
        eqn=l,
        style=r"#da93ea",
        label="Quad.",
        offset=[0, 80],
        use_all_known=False,
        guess={"a": 3.5e-7, "b": -4e-3, "c": 470},
        bounds=[
            {"name": "a", "min": 0, "max": 1},
            {"name": "b", "min": -1, "max": 1},
            {"name": "c", "min": 0, "max": 700},
        ],
        add_fit_params=True,
        granularity=10000,
        # gof=True,
        # scale_covar=True,
    )
    print(peaks)
    p = params

    ax.set_title(f"Quadratischer Wellenlängenverteilung über die Pixel")
    P.ax_legend_all(loc=1)
    ax = P.savefig(f"mappingPxToWaveLength.pdf")

    def pxToNm(x):
        return p["a"].value * (x - p["b"].value) ** 2 + p["c"].value

    def nmToPx(l):
        return np.sqrt((l - p["c"].value) / p["a"].value) + p["b"].value

    # lambda_violet = wavelength(violet)
    # print(f"Wavelength of violet line: {lambda_violet}")
    # print(f"Spectral resolution: {lambda_violet/0.4}")
    return (pxToNm, nmToPx)


def imgPrep() -> None:
    base_names = ["Hg_", "Halo_", "I_"]

    gm = {
        "I": r"I",
        "l": r"\lambda",
        "px": r"px",
        "a": r"a",
        "b": r"b",
        "c": r"c",
    }
    gv = {
        "I": r"relativ",
        "a": r"\si{\nano\meter\per\px\squared}",
        "b": r"\si{\px}",
        "c": r"\si{\nano\meter}",
        "l": r"\si{\nano\meter}",
        "px": r"\si{\px}",
    }

    pd.set_option("display.max_columns", None)
    plt.rcParams["axes.axisbelow"] = True
    P = Project("Spektrometer", global_variables=gv, global_mapping=gm, font=13)
    P.output_dir = "./"
    P.figure.set_size_inches((12, 6))

    for name in base_names:

        pics = list()
        peakspics = list()

        for picture_name in [f"{name}{i}.JPG" for i in range(8)]:

            I = cv2.imread(f"{PICTURE_DIR}{picture_name}")
            imm = I[y_start:y_end, x_start:x_end]

            # Gamma correction for low exposure
            gamma = 0.6
            lookUpTable = np.empty((1, 256), np.uint8)
            for i in range(256):
                lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

            gamma_pic = cv2.LUT(imm, lookUpTable).copy()
            pics.append(gamma_pic)
            # color_converted = cv2.cvtColor(gamma_pic, cv2.COLOR_BGR2RGB)
            # pil_image = Image.fromarray(color_converted)
            # YUV = cv2.cvtColor(I, cv2.COLOR_BGR2YUV)
            # YUV[:, :, 0] = cv2.equalizeHist(YUV[:, :, 0])
            # # convert the YUV image back to RGB format
            # img_output = cv2.cvtColor(YUV, cv2.COLOR_YUV2BGR)
            # cv2.imwrite(f"./cleared/{picture_name}", img_output)
            RGB = cv2.cvtColor(imm, cv2.COLOR_BGR2RGB)  # convert to RGB
            R, G, B = cv2.split(RGB)

            # Create a CLAHE object: The image is divided into small block 8x8 which they are equalized as usual.
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            # Applying this method to each channel of the color image
            output_2R = clahe.apply(R)
            output_2G = clahe.apply(G)
            output_2B = clahe.apply(B)

            # mergin each channel back to one
            img_output = cv2.merge((output_2R, output_2G, output_2B))
            eq = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)
            peakspics.append(eq)

        cv_panorama = cv2.flip(np.hstack(pics), 1)
        pan = cv2.cvtColor(cv_panorama, cv2.COLOR_BGR2RGB)
        peaks = cv2.flip(np.hstack(peakspics), 1)
        cv2.imwrite(f"./cleared/{name}comb.jpg", cv_panorama)
        cv2.imwrite(f"./cleared/{name}comb_eq.jpg", cv_panorama)
        analysis(P, peaks, pan, name)


if __name__ == "__main__":
    imgPrep()
