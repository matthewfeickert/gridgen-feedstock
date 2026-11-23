import ctypes
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


bindir = Path(sys.prefix) / "Library" / "bin"
with os.add_dll_directory(str(bindir)):
    library = ctypes.CDLL(str(bindir / "gridgen.dll"))
    # Keep in sync with the gridgen.def exports in CMakeLists.txt
    for symbol in (
        "gridgen_setverbose",
        "gridgen_printversion",
        "gridgen_printhelpalg",
        "gridgen_printhelpprm",
        "gridgen_generategrid",
        "gridgen_generategrid2",
        # gridutils API that pygridgen calls through gridgen's library
        "gridnodes_getx",
        "gridnodes_gety",
        "gridnodes_getnce1",
        "gridnodes_getnce2",
        "gridnodes_destroy",
        "gridmap_build",
    ):
        getattr(library, symbol)
    library.gridgen_setverbose.argtypes = [ctypes.c_int]
    library.gridgen_setverbose.restype = None
    library.gridgen_setverbose(0)

# Generate a grid through the library API, as pygridgen does.
c_double_p = ctypes.POINTER(ctypes.c_double)
c_int_p = ctypes.POINTER(ctypes.c_int)
c_double_pp = ctypes.POINTER(c_double_p)
# cf. the gridgen_generategrid2 prototype in gridgen.h
library.gridgen_generategrid2.argtypes = (
    [ctypes.c_int]  # nbdry
    + [c_double_p] * 3  # xbdry, ybdry, beta
    + [ctypes.c_int] * 4  # ul, nx, ny, ngrid
    + [c_double_p] * 2  # xgrid, ygrid
    + [ctypes.c_int] * 2  # nnodes, newton
    + [ctypes.c_double]  # precision
    + [ctypes.c_int] * 4  # checksimplepoly, thin, nppe, verbose
    + [c_int_p, c_double_pp]  # nsigmas, sigmas
    + [c_int_p, c_double_pp, c_double_pp]  # nrect, xrect, yrect
)
library.gridgen_generategrid2.restype = ctypes.c_void_p
library.gridnodes_getx.restype = ctypes.POINTER(c_double_p)
library.gridnodes_gety.restype = ctypes.POINTER(c_double_p)
for function in (
    library.gridnodes_getx,
    library.gridnodes_gety,
    library.gridnodes_destroy,
):
    function.argtypes = [ctypes.c_void_p]
library.gridnodes_destroy.restype = None


def generate_nodes(boundary, beta, nx, ny, xgrid=None, ygrid=None):
    """Generate a grid with gridgen_generategrid2 and return its nodes.

    Without xgrid and ygrid, the nodes are an nx by ny grid. With them, the
    nodes are the nx * ny points (xgrid, ygrid) of the unit square of the image
    region, as pygridgen.Focus passes them.
    """
    nbdry = len(boundary)
    DoubleArray = ctypes.c_double * nbdry
    ngrid = 0 if xgrid is None else nx * ny
    GridArray = ctypes.c_double * ngrid
    nodes = library.gridgen_generategrid2(
        nbdry,
        DoubleArray(*(x for x, _ in boundary)),
        DoubleArray(*(y for _, y in boundary)),
        DoubleArray(*beta),
        0,  # ul: index of the upper-left corner
        nx,
        ny,
        ngrid,
        None if xgrid is None else GridArray(*xgrid),
        None if ygrid is None else GridArray(*ygrid),
        14,  # nnodes
        1,  # newton
        1e-12,  # precision
        1,  # checksimplepoly
        1,  # thin
        3,  # nppe
        0,  # verbose
        # nsigmas, sigmas, nrect, xrect, yrect: outputs that are not checked
        ctypes.byref(ctypes.c_int(0)),
        ctypes.byref(c_double_p()),
        ctypes.byref(ctypes.c_int(0)),
        ctypes.byref(c_double_p()),
        ctypes.byref(c_double_p()),
    )
    assert nodes, "gridgen_generategrid2 returned NULL"
    # gridnodes stores the nodes contiguously, row after row
    xs = library.gridnodes_getx(nodes)[0]
    ys = library.gridnodes_gety(nodes)[0]
    points = [(xs[i], ys[i]) for i in range(nx * ny)]
    library.gridnodes_destroy(nodes)
    return points


boundary = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
nx, ny = 41, 21
# beta: four 90 degree corners
points = generate_nodes(boundary, [1.0] * 4, nx, ny)
assert all(math.isfinite(x) and math.isfinite(y) for x, y in points)
assert all(-1e-9 <= x <= 2.0 + 1e-9 and -1e-9 <= y <= 1.0 + 1e-9 for x, y in points)
corners = [points[i] for i in (0, nx - 1, (ny - 1) * nx, nx * ny - 1)]
for x_expected, y_expected in boundary:
    assert any(
        math.isclose(x, x_expected, abs_tol=1e-6)
        and math.isclose(y, y_expected, abs_tol=1e-6)
        for x, y in corners
    ), corners

# The boundary of pygridgen's grid tests (pygridgen 0.3.0 tests/test_grid.py)
# splits into more than one quadrilateral, so map_quadrilaterals looks up its
# internal edges, which crashed with -DNDEBUG before
# patches/do-not-insert-into-hash-table-inside-assert.patch. Also cover the
# focused (ngrid > 0) path that pygridgen uses.
pentagon = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.5), (1.0, 1.0), (0.0, 1.0)]
nx, ny = 5, 10
focus_x = [i / (nx - 1) for j in range(ny) for i in range(nx)]
focus_y = [j / (ny - 1) for j in range(ny) for i in range(nx)]
for xgrid, ygrid in ((None, None), (focus_x, focus_y)):
    points = generate_nodes(pentagon, [1.0, 1.0, 0.0, 1.0, 1.0], nx, ny, xgrid, ygrid)
    # Nodes outside the polygon are NaN, so only check the others.
    inside = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
    assert inside, points
    assert all(
        -1e-9 <= x <= 2.0 + 1e-9 and -1e-9 <= y <= 1.0 + 1e-9 for x, y in inside
    ), inside

# Exercise Triangle and complex arithmetic with a small rectangular grid.
with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    (work / "boundary.txt").write_text(
        "0 0 1\n1 0 1\n1 1 1\n0 1 1\n", encoding="ascii"
    )
    (work / "grid.prm").write_text(
        "input boundary.txt\noutput grid.txt\nnx 5\nny 5\n", encoding="ascii"
    )
    subprocess.run(
        [str(bindir / "gridgen.exe"), "grid.prm"],
        cwd=work,
        check=True,
        timeout=60,
    )
    lines = (work / "grid.txt").read_text().splitlines()
    coordinates = [line.split() for line in lines if not line.startswith("##")]
    assert len(coordinates) == 25, lines
    assert all(
        len(point) == 2 and all(math.isfinite(float(value)) for value in point)
        for point in coordinates
    ), lines
