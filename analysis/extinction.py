# pyright: reportUnboundVariable=false
# pyright: reportUndefinedVariable=false

from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy.constants import speed_of_light, h, elementary_charge
from scipy.signal import find_peaks
from imageprep import genBasisMap
from numpy.typing import NDArray
from labtool_ex2 import Project
from uncertainties import ufloat

DATAPATH = "./data/"
OUTPUTPATH = "./output/"
PICTURE_DIR = f"{DATAPATH}edited/"

pxToNm = callable
nmToPx = callable


def readIn(path, skiprows=0):
    lines = []
    with open(path, "r", encoding="ISO-8859-1") as f:
        lines = f.readlines()
    if skiprows:
        lines = lines[skiprows:]

    L, I = [], []
    for str_ in lines:
        try:
            str_ = str_.replace(",", ".")
            l, i = str_.split("\t")
            L.append(float(l))
            I.append(float(i))
        except:
            pass
    return np.asarray(L), np.asarray(I)


def extinction(val: NDArray, ref: NDArray) -> NDArray:
    return np.log10(val / ref)


def plot_extinction(P: Project) -> None:
    P.figure.clear()
    P.data = pd.DataFrame(None)
    ax = P.figure.add_subplot()

    # Extinktion prism
    with open(f"{DATAPATH}Halo_pickle", "rb") as file:
        I_ref = pickle.load(file)  # Halogen
    with open(f"{DATAPATH}I_pickle", "rb") as file:
        I_Iod = pickle.load(file)
    Ext = extinction(I_Iod, I_ref)
    P.data["E"] = Ext
    pixels = np.arange(len(Ext))
    _l = pxToNm(pixels)
    P.data["l"] = _l
    P.vload()

    P.plot(
        axes=ax,
        x=l,
        y=E,
        label="Prismenspektrograph",
        style="#da93ea",
    )

    ax.set_title("Extinktionsspektrum einer Iod-Zelle vor einer Halogenlampe")
    P.ax_legend_all(loc=0)
    P.figure.tight_layout(pad=1)
    ax = P.savefig(f"prism_extinction.pdf", clear=True)

    low = int(nmToPx(508))
    high = int(nmToPx(550))
    _l = pxToNm(np.arange(len(Ext)))
    l_band = _l[low:high]
    E_band = -Ext[low:high]
    peaks, *_ = find_peaks(E_band, distance=130, prominence=0.01)
    l_peaks, E_peaks = l_band[peaks], E_band[peaks]
    P.data = pd.DataFrame(None)
    P.data["E"] = E_band
    P.data["l"] = l_band
    P.plot(
        axes=ax,
        x=l,
        y=E,
        label="Prismenspektrograph",
        style="#da93ea",
    )
    P.data = pd.DataFrame(None)
    P.data["E"] = E_peaks
    P.data["l"] = l_peaks
    P.plot_data(
        axes=ax,
        x=l,
        y=E,
        label="Peaks",
        style="#a30000",
    )
    ax.set_title("Extinktionsspektrum einer Iod-Zelle vor einer Halogenlampe")
    P.ax_legend_all(loc=0)
    P.figure.tight_layout(pad=1)
    ax = P.savefig(f"prism_ausschnitt_extinction.pdf", clear=True)

    waveNumber(P, l_peaks / 10e6, "Prisma")
    P.figure.clear()
    P.data = pd.DataFrame(None)
    ax = P.figure.add_subplot()

    _l, halogen = readIn(f"{PICTURE_DIR}halogen.txt", skiprows=17)
    _, HG = readIn(f"{PICTURE_DIR}quecksilber.txt", skiprows=17)
    _, Iod = readIn(f"{PICTURE_DIR}iod.txt", skiprows=17)

    Ext = -extinction(Iod, halogen)
    P.data = pd.DataFrame(None)
    P.data["E"] = Ext
    P.data["l"] = _l
    P.plot(
        axes=ax,
        x=l,
        y=E,
        label="Gitterspektrograph",
        style="#da93ea",
    )
    ax.set_title("Extinktionsspektrum einer Iod-Zelle vor einer Halogenlampe")
    P.ax_legend_all(loc=1)
    P.figure.tight_layout(pad=1)
    ax = P.savefig(f"gitter_extinction.pdf", clear=True)

    filter = np.logical_and((550 < _l), (_l < 615))
    l_band = _l[filter]
    E_band = Ext[filter]
    peaks, *_ = find_peaks(E_band, distance=11, prominence=0.001)
    l_peaks, E_peaks = l_band[peaks], E_band[peaks]
    P.data = pd.DataFrame(None)
    P.data["E"] = E_band
    P.data["l"] = l_band
    P.plot(
        axes=ax,
        x=l,
        y=E,
        label="Gitterspektrograph",
        style="#da93ea",
    )
    P.data = pd.DataFrame(None)
    P.data["E"] = E_peaks
    P.data["l"] = l_peaks
    P.plot_data(
        axes=ax,
        x=l,
        y=E,
        label="Peaks",
        style="#a30000",
    )
    ax.set_title("Extinktionsspektrum einer Iod-Zelle vor einer Halogenlampe")
    P.ax_legend_all(loc=0)
    P.figure.tight_layout(pad=1)
    ax = P.savefig(f"gitter_ausschnitt_extinction.pdf", clear=True)
    waveNumber(P, l_peaks / 10e6, "Gitter")
    P.figure.clear()
    P.data = pd.DataFrame(None)
    ax = P.figure.add_subplot()

    max_val = max(halogen.max(), HG.max(), Iod.max())
    P.data["l"] = _l
    P.data["I"] = halogen / max_val
    # ax.plot(_l, halogen/max_val, label="Halogen")
    P.plot(
        axes=ax,
        x=l,
        y=I,
        label="Halogenlampe",
    )

    ax.plot(_l, HG / max_val, label="Quecksilberlampe")
    ax.plot(_l, Iod / max_val, label=r"Halogen mit Iodrohr")
    ax.set_title("Intensitätskurve den ver. Proben (Gitterspektrograph)")
    P.ax_legend_all(loc=0)
    P.figure.set_size_inches((11, 6))
    ax = P.savefig(f"intensity_spektrum_gitter.pdf", clear=True)


def waveNumber(P: Project, peaks, name: str):
    P.figure.clear()
    P.data = pd.DataFrame(None)
    ax = P.figure.add_subplot()
    P.data["v"] = 1 / peaks
    P.data["i"] = np.arange(len(P.data["v"]))
    P.vload()

    P.plot_data(
        axes=ax,
        x=i,
        y=v,
        label=f"Referenzwerte {name}",
        style="#a30000",
    )

    v = d * (i - e) ** 2 + f

    params = P.plot_fit(
        axes=ax,
        x=i,
        y=v,
        eqn=v,
        style=r"#da93ea",
        label="Quad.",
        offset=[0, 10],
        use_all_known=False,
        guess={"d": -1, "e": -52, "f": 20000},
        bounds=[
            {"name": "d", "min": -20, "max": 400},
            {"name": "e", "min": -70, "max": 100},
            {"name": "f", "min": 0, "max": 30000},
        ],
        add_fit_params=True,
        granularity=10000,
        # gof=True,
        # scale_covar=True,
    )

    def quad(x):
        return ufloat(params["d"].value, params["d"].stderr) * (
            x - ufloat(params["e"].value, params["e"].stderr)
        ) ** 2 + ufloat(params["f"].value, params["f"].stderr)

    ax.set_title("Quadratische Abhängigkeit der Wellenzahlen")
    P.ax_legend_all(loc=1)
    P.figure.tight_layout(pad=1)
    ax = P.savefig(f"waveNumberFit{name}.pdf", clear=True)

    # Wavenumber prism difference
    v1 = P.data["v"].values[:-1]
    v2 = P.data["v"].values[1:]
    P.data = pd.DataFrame(None)
    P.data["Dv"] = v2 - v1
    P.data["i"] = np.arange(len(P.data["Dv"]))
    P.vload()
    P.plot_data(
        axes=ax,
        x=i,
        y=Dv,
        label=f"Referenzwerte {name}",
        style="#a30000",
    )

    Dv = k * i + d

    pp = P.plot_fit(
        axes=ax,
        x=i,
        y=Dv,
        eqn=Dv,
        style=r"#da93ea",
        label="Lin.",
        offset=[0, 10],
        use_all_known=False,
        guess={"k": -2, "d": -52},
        bounds=[
            {"name": "k", "min": -4, "max": 0},
            {"name": "d", "min": -102, "max": 1},
        ],
        add_fit_params=True,
        granularity=10000,
        # gof=True,
        # scale_covar=True,
    )

    def lin(x):
        return ufloat(pp["k"].value, pp["k"].stderr) * x + ufloat(
            pp["d"].value, pp["d"].stderr
        )

    def invlin(v):
        return (v - ufloat(pp["d"].value, pp["d"].stderr)) / ufloat(
            pp["k"].value, pp["k"].stderr
        )

    ax.set_title("Lineare Abhängigkeit der Wellenzahlenabstände")
    P.ax_legend_all(loc=1)
    P.figure.tight_layout(pad=1)
    ax = P.savefig(f"waveNumberDeltasFit{name}.pdf", clear=True)

    # Calculation
    vindex = invlin(0)
    v = quad(vindex) * 1e2  # To cm to m SI
    print("----------------------------------")
    print(f"{vindex=}")
    print(f"{v=}")
    wavelength = 1 / v
    print(f"{wavelength=}")

    print("Scheitelenergie")
    E = h * speed_of_light * v
    E_ev = E / elementary_charge
    E_diss = E_ev - ufloat(0.970, 0.005)  # Anregungsenergie
    print(f"{E=}", f"{E_ev=}", f"{E_diss=}")
    print("----------------------------------")


if __name__ == "__main__":

    gm = {
        "I": r"I",
        "i": r"\text{Index}",
        "l": r"\lambda",
        "px": r"px",
        "a": r"a",
        "b": r"b",
        "c": r"c",
        "d": r"a",
        "f": r"b",
        "e": r"c",
        "k": r"k",
        "v": r"\nu",
        "Dv": r"\Delta\nu",
        "E": r"E",
    }
    gv = {
        "I": r"relativ",
        "i": r"1",
        "a": r"\si{\nano\meter\per\px\squared}",
        "b": r"\si{\px}",
        "c": r"\si{\nano\meter}",
        "d": r"\si{\per\centi\meter}",
        "f": r"\si{\per\centi\meter}",
        "k": r"\si{\per\centi\meter}",
        "e": r"1",
        "v": r"\si{\per\centi\meter}",
        "Dv": r"\si{\per\centi\meter}",
        "l": r"\si{\nano\meter}",
        "px": r"\si{\px}",
        "E": r"1",
    }

    pd.set_option("display.max_columns", None)
    plt.rcParams["axes.axisbelow"] = True
    P = Project("Spektrometer", global_variables=gv, global_mapping=gm, font=13)
    P.output_dir = "./"
    P.figure.set_size_inches((11, 3))
    pxToNm, nmToPx = genBasisMap(P)
    plot_extinction(P)
