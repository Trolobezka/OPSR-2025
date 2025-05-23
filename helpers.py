import math
import typing as T

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import sympy as smp
from IPython.display import Latex, display
from sympy.core._print_helpers import Printable  # type: ignore

HALF = smp.Rational("0.5")
GRAVITY = smp.Rational("9.81")


def print2(*objs: Printable | str, newline: bool = False) -> None:
    if len(objs) == 0:
        return
    desc = ""
    if isinstance(objs[0], str):
        desc = objs[0] + r": $\ \ $ "
        if newline:
            desc += r"$\newline$ "
        objs = objs[1:]
    display(
        Latex(
            desc
            + r"$\displaystyle "
            + (r",\newline " if newline else r",\ ").join(
                [(("{" if i % 2 == 0 else r"\textcolor{gray}{") + smp.latex(obj) + "}") for (i, obj) in enumerate(objs)]
            )
            + "$"
        )
    )


def rk4(
    f_func: T.Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64], float],
        npt.NDArray[np.float64],
    ],
    u_func: T.Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]],
    x0: npt.NDArray[np.float64],
    n: int,
    m: int,
    dt: float,
    t0: float,
    t1: float,
) -> T.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    ```
    # Dynamics function signature
    def f_func(
        x: npt.NDArray[np.float64],     # (n,)
        u: npt.NDArray[np.float64],     # (m,)
        t: float
    ) -> npt.NDArray[np.float64]:       # (n,)
        ...

    # Control function signature
    def u_func(
        x: npt.NDArray[np.float64],     # (n,)
        t: float
    ) -> npt.NDArray[np.float64]:       # (m,)
        ...
    ```

    Returns a tuple of:
    - `t`: time vector `(N + 1,)`
    - `x`: state vector `(N + 1, n)`
    - `u`: control vector `(N, m)`

    where `N = math.floor((t1 - t0) / dt)`.
    """
    N = math.floor((t1 - t0) / dt)
    t = np.linspace(t0, t1, N + 1, dtype=np.float64)
    x = np.zeros((N + 1, n), dtype=np.float64)
    u = np.zeros((N, m), dtype=np.float64)
    x[0] = x0
    for i in range(N):
        u[i] = u_func(x[i], t[i])
        k1 = f_func(x[i], u[i], t[i])
        k2 = f_func(x[i] + 0.5 * dt * k1, u[i], t[i] + 0.5 * dt)
        k3 = f_func(x[i] + 0.5 * dt * k2, u[i], t[i] + 0.5 * dt)
        k4 = f_func(x[i] + dt * k3, u[i], t[i] + dt)
        x[i + 1] = x[i] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, x, u


def discretization_by_rk4_integration(
    t: float,
    x: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    params: T.Dict[str, T.Any],
) -> npt.NDArray[np.float64]:
    _f_func = params["f_func"]
    _dt = params["dt"]
    _dt_internal = params["dt_internal"]
    _solution = rk4(
        f_func=_f_func,
        u_func=lambda _x, _t: u.ravel(),
        x0=x.ravel(),
        n=6,
        m=2,
        dt=_dt_internal,
        t0=t,
        t1=t + _dt + 1e-6,
    )
    _t1 = _solution[0][-1]
    assert abs(_t1 - t - _dt) < (_dt_internal * 0.1), f"time {_t1}"
    _x = _solution[1][-1, :]
    return _x


def _plot(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    label: str | None,
    xlabel: str | None,
    ylabel: str | None,
):
    plt.plot(x, y, label=label)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.grid(True, "both")
    if label is not None:
        plt.legend()


def plot(
    time_1: npt.NDArray[np.float64],
    x_c: npt.NDArray[np.float64] | None,
    y_c: npt.NDArray[np.float64] | None,
    alpha: npt.NDArray[np.float64] | None,
    dx_c: npt.NDArray[np.float64] | None,
    dy_c: npt.NDArray[np.float64] | None,
    dalpha: npt.NDArray[np.float64] | None,
    label: str | None = None,
    rows: int = 2,
):
    if x_c is not None:
        plt.subplot(rows, 3, 1)
        _plot(time_1, x_c, label, r"time $t$ [s]", r"position $x_c$ [m]")
    if y_c is not None:
        plt.subplot(rows, 3, 2)
        _plot(time_1, y_c, label, r"time $t$ [s]", r"position $y_c$ [m]")
    if alpha is not None:
        plt.subplot(rows, 3, 3)
        _plot(time_1, np.degrees(alpha), label, r"time $t$ [s]", r" rotation $\alpha$ [deg]")
    if dx_c is not None:
        plt.subplot(rows, 3, 4)
        _plot(time_1, dx_c, label, r"time $t$ [s]", r"velocity $dx_c / dt$ [m/s]")
    if dy_c is not None:
        plt.subplot(rows, 3, 5)
        _plot(time_1, dy_c, label, r"time $t$ [s]", r"velocity $dy_c / dt$ [m/s]")
    if dalpha is not None:
        plt.subplot(rows, 3, 6)
        _plot(time_1, np.degrees(dalpha), label, r"time $t$ [s]", r"velocity $d\alpha / dt$ [deg/s]")


def plot2(
    time_1: npt.NDArray[np.float64],
    x_c: npt.NDArray[np.float64] | None,
    y_c: npt.NDArray[np.float64] | None,
    alpha: npt.NDArray[np.float64] | None,
    dx_c: npt.NDArray[np.float64] | None,
    dy_c: npt.NDArray[np.float64] | None,
    dalpha: npt.NDArray[np.float64] | None,
    f_x: npt.NDArray[np.float64] | None,
    f_y: npt.NDArray[np.float64] | None,
    label: str | None = None,
):
    plot(time_1, x_c, y_c, alpha, dx_c, dy_c, dalpha, label=label, rows=3)
    if f_x is not None:
        plt.subplot(3, 3, 7)
        _plot(time_1, f_x, label, r"time $t$ [s]", r"force $f_x$ [N]")
    if f_y is not None:
        plt.subplot(3, 3, 8)
        _plot(time_1, f_y, label, r"time $t$ [s]", r"force $f_y$ [N]")


def print_array(array: npt.NDArray[np.float64]) -> None:
    print(
        np.array2string(
            array,
            max_line_width=1000,
            precision=6,
            suppress_small=True,
        )
    )
