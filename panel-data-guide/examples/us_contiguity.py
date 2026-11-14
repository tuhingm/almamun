"""Land-border (rook) contiguity of the 48 contiguous U.S. states, with the state names used in produc.csv.

    W = contiguity_matrix(names)      row-standardised weights matrix (rows sum to one), states in the order of `names`
"""
import numpy as np

_B = """AL: FL GA MS TN
AZ: CA NV NM UT
AR: LA MO MS OK TN TX
CA: AZ NV OR
CO: KS NE NM OK UT WY
CT: MA NY RI
DE: MD NJ PA
FL: AL GA
GA: AL FL NC SC TN
ID: MT NV OR UT WA WY
IL: IN IA KY MO WI
IN: IL KY MI OH
IA: IL MN MO NE SD WI
KS: CO MO NE OK
KY: IL IN MO OH TN VA WV
LA: AR MS TX
ME: NH
MD: DE PA VA WV
MA: CT NH NY RI VT
MI: IN OH WI
MN: IA ND SD WI
MS: AL AR LA TN
MO: AR IL IA KS KY NE OK TN
MT: ID ND SD WY
NE: CO IA KS MO SD WY
NV: AZ CA ID OR UT
NH: ME MA VT
NJ: DE NY PA
NM: AZ CO OK TX
NY: CT MA NJ PA VT
NC: GA SC TN VA
ND: MN MT SD
OH: IN KY MI PA WV
OK: AR CO KS MO NM TX
OR: CA ID NV WA
PA: DE MD NJ NY OH WV
RI: CT MA
SC: GA NC
SD: IA MN MT NE ND WY
TN: AL AR GA KY MS MO NC VA
TX: AR LA NM OK
UT: AZ CO ID NV WY
VT: MA NH NY
VA: KY MD NC TN WV
WA: ID OR
WV: KY MD OH PA VA
WI: IL IA MI MN
WY: CO ID MT NE SD UT"""

_NAMES = {"AL": "ALABAMA", "AZ": "ARIZONA", "AR": "ARKANSAS", "CA": "CALIFORNIA", "CO": "COLORADO", "CT": "CONNECTICUT", "DE": "DELAWARE",
          "FL": "FLORIDA", "GA": "GEORGIA", "ID": "IDAHO", "IL": "ILLINOIS", "IN": "INDIANA", "IA": "IOWA", "KS": "KANSAS", "KY": "KENTUCKY",
          "LA": "LOUISIANA", "ME": "MAINE", "MD": "MARYLAND", "MA": "MASSACHUSETTS", "MI": "MICHIGAN", "MN": "MINNESOTA", "MS": "MISSISSIPPI",
          "MO": "MISSOURI", "MT": "MONTANA", "NE": "NEBRASKA", "NV": "NEVADA", "NH": "NEW_HAMPSHIRE", "NJ": "NEW_JERSEY", "NM": "NEW_MEXICO",
          "NY": "NEW_YORK", "NC": "NORTH_CAROLINA", "ND": "NORTH_DAKOTA", "OH": "OHIO", "OK": "OKLAHOMA", "OR": "OREGON", "PA": "PENNSYLVANIA",
          "RI": "RHODE_ISLAND", "SC": "SOUTH_CAROLINA", "SD": "SOUTH_DAKOTA", "TN": "TENNESSE", "TX": "TEXAS", "UT": "UTAH", "VT": "VERMONT",
          "VA": "VIRGINIA", "WA": "WASHINGTON", "WV": "WEST_VIRGINIA", "WI": "WISCONSIN", "WY": "WYOMING"}   # "TENNESSE" as spelled in produc.csv


def neighbours():
    nb = {}
    for line in _B.splitlines():
        a, rest = line.split(":")
        nb[_NAMES[a.strip()]] = [_NAMES[x] for x in rest.split()]
    for a, bs in nb.items():                      # the list must be symmetric
        for b in bs:
            assert a in nb[b], (a, b)
    return nb


def contiguity_matrix(names, standardise=True):
    nb = neighbours(); idx = {n: i for i, n in enumerate(names)}
    A = np.zeros((len(names), len(names)))
    for a, bs in nb.items():
        for b in bs:
            A[idx[a], idx[b]] = 1.0
    return A / A.sum(1, keepdims=True) if standardise else A
