"""
PINN para difusión de H2 en Polietileno (PE)
------------------------------------------------
- Ecuación:  \partial_t C(x,t) = D(x) \partial_{xx} C(x,t) + S(C,x,t) (por defecto S=0)
- Dominio 1D: x \in [0, L], t \in [0, T]
- Modos: forward (predicción) e inverse (identificación de parámetros)
- Opción multicapa con interfaz en x = x_if: D(x) = D1 en [0,x_if], D2 en (x_if,L]
  Condiciones de interfaz:
    * Continuidad de concentración: C1(x_if,t) = C2(x_if,t)
    * Continuidad de flujo: D1 * \partial_x C1(x_if,t) = D2 * \partial_x C2(x_if,t)
- Optimizadores: Adam + L-BFGS
- Entradas: (x,t) normalizados, salida: C(x,t)
- Integración de datos: loss_data con puntos (x,t,C) provenientes de mediciones/CSV

Notas de integración con tu repositorio:
- Carga de datos experimentales/sintéticos: ver función `load_data_csv`.
- Ajusta la malla de evaluación y exportación en `save_eval_grid`.

Autor: ChatGPT (para ing)
"""

from __future__ import annotations
import os
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =========================================
# 1) Configuración
# =========================================

@dataclass
class PINNConfig:
    L: float = 1.0                  # Longitud [m]
    T: float = 1.0                  # Tiempo total [s]

    # Difusividades (m^2/s)
    use_bilayer: bool = False       # True: usa D1 y D2 con interfaz
    x_if: float = 0.5               # posición de la interfaz (0 < x_if < L)
    D1_init: float = 1e-10          # inicial para capa 1
    D2_init: float = 5e-11          # inicial para capa 2
    D_const_init: float = 8e-11     # si no hay interfaz

    train_inverse: bool = False     # si True, estima D (o D1,D2) como parámetros entrenables

    # Condiciones de frontera (ejemplo):
    # C(0,t) = C_left(t) (Dirichlet), \partial_x C(L,t) = 0 (Neumann cero)
    use_dirichlet_left: bool = True
    C_left_value: float = 1.0       # concentración impuesta en x=0
    use_neumann_right: bool = True

    # Condición inicial: C(x,0) = C0(x)
    C0_value: float = 0.0

    # Núm. de puntos de entrenamiento
    N_f: int = 4000                 # collocation PDE
    N_bc_t: int = 800               # puntos en frontera temporal (para BC de x)
    N_ic_x: int = 800               # puntos para condición inicial
    N_if_t: int = 800               # puntos en interfaz (si bilayer)
    N_data: int = 0                 # puntos de datos (si hay CSV)

    # Pesos de pérdidas
    w_pde: float = 1.0
    w_ic: float = 1.0
    w_bc: float = 1.0
    w_if: float = 1.0               # interfaz (bilayer)
    w_data: float = 1.0

    # Red
    layers: Tuple[int, ...] = (2, 128, 128, 128, 128, 1)
    act: str = "tanh"               # tanh | relu | gelu | silu

    # Entrenamiento
    lr: float = 1e-3
    adam_steps: int = 15000
    use_lbfgs: bool = True
    lbfgs_max_iter: int = 5000
    seed: int = 42

    # Exportación / evaluación
    grid_nx: int = 201
    grid_nt: int = 101
    out_dir: str = "outputs_pinn"
    out_tag: str = "pe_h2"


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================
# 2) Utilidades de muestreo y normalización
# =========================================

def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def latin_hypercube(n: int, dims: int) -> np.ndarray:
    # LHS simple en [0,1]^dims
    seg = np.linspace(0, 1, n + 1)
    pts = np.zeros((n, dims))
    for d in range(dims):
        low = seg[:-1]
        high = seg[1:]
        pts[:, d] = np.random.uniform(low, high)
        np.random.shuffle(pts[:, d])
    return pts


# =========================================
# 3) Red MLP
# =========================================

class MLP(nn.Module):
    def __init__(self, layers: Tuple[int, ...], act: str = "tanh"):
        super().__init__()
        acts = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        self.act = acts.get(act, nn.Tanh())
        net = []
        for i in range(len(layers) - 2):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            net.append(self.act)
        net.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*net)

        # Xavier init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# =========================================
# 4) Modelo PINN
# =========================================

class DiffusionPINN(nn.Module):
    def __init__(self, cfg: PINNConfig):
        super().__init__()
        self.cfg = cfg
        self.model = MLP(cfg.layers, cfg.act)

        # Parámetros difusivos (entrenables si train_inverse)
        if cfg.use_bilayer:
            self.logD1 = nn.Parameter(torch.log(torch.tensor(cfg.D1_init)))
            self.logD2 = nn.Parameter(torch.log(torch.tensor(cfg.D2_init)))
        else:
            self.logD = nn.Parameter(torch.log(torch.tensor(cfg.D_const_init)))

        if not cfg.train_inverse:
            # Congelar si no se estiman
            if cfg.use_bilayer:
                self.logD1.requires_grad_(False)
                self.logD2.requires_grad_(False)
            else:
                self.logD.requires_grad_(False)

    def D(self, x: torch.Tensor):
        """ Difusividad pieza a pieza (si bilayer) o constante. x en [0, L]. """
        if self.cfg.use_bilayer:
            D1 = torch.exp(self.logD1)
            D2 = torch.exp(self.logD2)
            return torch.where(x <= self.cfg.x_if, D1, D2)
        else:
            return torch.exp(self.logD)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # Entrada en [0,1] normalizada: x/L, t/T
        xin = x / self.cfg.L
        tin = t / self.cfg.T
        X = torch.cat([xin, tin], dim=1)
        C = self.model(X)
        return C

    def pde_residual(self, x: torch.Tensor, t: torch.Tensor):
        x.requires_grad_(True)
        t.requires_grad_(True)
        C = self.forward(x, t)
        dC_dt = torch.autograd.grad(C, t, grad_outputs=torch.ones_like(C), create_graph=True)[0]
        dC_dx = torch.autograd.grad(C, x, grad_outputs=torch.ones_like(C), create_graph=True)[0]
        d2C_dx2 = torch.autograd.grad(dC_dx, x, grad_outputs=torch.ones_like(dC_dx), create_graph=True)[0]

        Dxt = self.D(x)
        # Fuente opcional (S=0 por defecto)
        S = 0.0
        res = dC_dt - Dxt * d2C_dx2 - S
        return res

    def grad_x(self, x: torch.Tensor, t: torch.Tensor):
        x.requires_grad_(True)
        t.requires_grad_(True)
        C = self.forward(x, t)
        dC_dx = torch.autograd.grad(C, x, grad_outputs=torch.ones_like(C), create_graph=True)[0]
        return dC_dx


# =========================================
# 5) Generación de datasets de entrenamiento
# =========================================

def sample_collocation(cfg: PINNConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    pts = latin_hypercube(cfg.N_f, 2)  # (x,t) en [0,1]
    x = to_tensor(pts[:, [0]]) * cfg.L
    t = to_tensor(pts[:, [1]]) * cfg.T
    return x, t

def sample_initial(cfg: PINNConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = to_tensor(np.random.rand(cfg.N_ic_x, 1)) * cfg.L
    t = torch.zeros_like(x)
    C = torch.full_like(x, cfg.C0_value)
    return x, t, C

def sample_bc_left(cfg: PINNConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = to_tensor(np.random.rand(cfg.N_bc_t, 1)) * cfg.T
    x = torch.zeros_like(t)
    C = torch.full_like(t, cfg.C_left_value)
    return x, t, C

def sample_bc_right_times(cfg: PINNConfig) -> torch.Tensor:
    return to_tensor(np.random.rand(cfg.N_bc_t, 1)) * cfg.T


def sample_interface_times(cfg: PINNConfig) -> torch.Tensor:
    return to_tensor(np.random.rand(cfg.N_if_t, 1)) * cfg.T


# =========================================
# 6) Carga de datos (opcional)
# =========================================

def load_data_csv(csv_path: str, cfg: PINNConfig, nmax: Optional[int] = None):
    """ Espera columnas: x,t,C  (en unidades del problema). """
    if not os.path.isfile(csv_path):
        return None
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if nmax is not None:
        arr = arr[:nmax]
    x = to_tensor(arr[:, [0]])
    t = to_tensor(arr[:, [1]])
    C = to_tensor(arr[:, [2]])
    return x, t, C


# =========================================
# 7) Pérdida total
# =========================================

def compute_losses(model: DiffusionPINN,
                   cfg: PINNConfig,
                   f_x: torch.Tensor, f_t: torch.Tensor,
                   ic: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                   bc_left: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                   bc_right_times: Optional[torch.Tensor] = None,
                   if_times: Optional[torch.Tensor] = None,
                   data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None):

    mse = nn.MSELoss()

    # PDE residual
    res = model.pde_residual(f_x, f_t)
    loss_pde = torch.mean(res**2)

    # IC: C(x,0) = C0(x)
    ic_x, ic_t, ic_C = ic
    pred_ic = model(ic_x, ic_t)
    loss_ic = mse(pred_ic, ic_C)

    # BC izquierda: Dirichlet C(0,t)=C_left
    loss_bc = torch.tensor(0.0)
    if bc_left is not None and cfg.use_dirichlet_left:
        bx, bt, bC = bc_left
        pred_left = model(bx, bt)
        loss_bc = mse(pred_left, bC)

    # BC derecha: Neumann \partial_x C(L,t) = 0
    if bc_right_times is not None and cfg.use_neumann_right:
        bt = bc_right_times
        bx = torch.full_like(bt, cfg.L)
        dCdx_right = model.grad_x(bx, bt)
        loss_bc = loss_bc + mse(dCdx_right, torch.zeros_like(dCdx_right))

    # Interfaz bilayer
    loss_if = torch.tensor(0.0)
    if cfg.use_bilayer and if_times is not None:
        t_if = if_times
        x_if_left = torch.full_like(t_if, cfg.x_if - 1e-6)
        x_if_right = torch.full_like(t_if, cfg.x_if + 1e-6)

        C_left = model(x_if_left, t_if)
        C_right = model(x_if_right, t_if)
        # Continuidad de concentración
        loss_if_c = mse(C_left, C_right)
        # Continuidad de flujo
        dCdx_left = model.grad_x(x_if_left, t_if)
        dCdx_right = model.grad_x(x_if_right, t_if)
        D1 = torch.exp(model.logD1)
        D2 = torch.exp(model.logD2)
        loss_if_flux = mse(D1 * dCdx_left, D2 * dCdx_right)
        loss_if = loss_if_c + loss_if_flux

    # Datos (x,t,C) opcionales
    loss_data = torch.tensor(0.0)
    if data is not None:
        dx, dt, dC = data
        pred_data = model(dx, dt)
        loss_data = mse(pred_data, dC)

    loss_total = (
        cfg.w_pde * loss_pde +
        cfg.w_ic * loss_ic +
        cfg.w_bc * loss_bc +
        cfg.w_if * loss_if +
        cfg.w_data * loss_data
    )

    losses = {
        'total': loss_total,
        'pde': loss_pde.detach(),
        'ic': loss_ic.detach(),
        'bc': loss_bc.detach(),
        'if': loss_if.detach(),
        'data': loss_data.detach(),
    }
    return loss_total, losses


# =========================================
# 8) Entrenamiento
# =========================================

def train_pinn(cfg: PINNConfig,
               data_csv: Optional[str] = None,
               device: str = "cpu"):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    model = DiffusionPINN(cfg).to(device)

    # Muestreos
    f_x, f_t = sample_collocation(cfg)
    ic = sample_initial(cfg)
    bc_left = sample_bc_left(cfg) if cfg.use_dirichlet_left else None
    bc_right_times = sample_bc_right_times(cfg) if cfg.use_neumann_right else None
    if_times = sample_interface_times(cfg) if cfg.use_bilayer else None

    if data_csv:
        data = load_data_csv(data_csv, cfg, nmax=cfg.N_data if cfg.N_data > 0 else None)
    else:
        data = None

    # A dispositivos
    f_x, f_t = f_x.to(device), f_t.to(device)
    ic = tuple(t.to(device) for t in ic)
    if bc_left is not None:
        bc_left = tuple(t.to(device) for t in bc_left)
    if bc_right_times is not None:
        bc_right_times = bc_right_times.to(device)
    if if_times is not None:
        if_times = if_times.to(device)
    if data is not None:
        data = tuple(t.to(device) for t in data)

    # Optimizadores
    params = list(model.model.parameters())
    if cfg.train_inverse:
        if cfg.use_bilayer:
            params += [model.logD1, model.logD2]
        else:
            params += [model.logD]

    optimizer = optim.Adam(params, lr=cfg.lr)

    def closure():
        optimizer.zero_grad()
        loss, _ = compute_losses(model, cfg, f_x, f_t, ic, bc_left, bc_right_times, if_times, data)
        loss.backward()
        return loss

    # Adam
    t0 = time.time()
    for step in range(1, cfg.adam_steps + 1):
        loss, logs = compute_losses(model, cfg, f_x, f_t, ic, bc_left, bc_right_times, if_times, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0 or step == 1:
            D_report = None
            if cfg.use_bilayer:
                D_report = (torch.exp(model.logD1).item(), torch.exp(model.logD2).item())
            else:
                D_report = (torch.exp(model.logD).item(),)
            print(f"[Adam {step:6d}] loss={loss.item():.3e} pde={logs['pde']:.2e} ic={logs['ic']:.2e} bc={logs['bc']:.2e} if={logs['if']:.2e} data={logs['data']:.2e} D={D_report}")

    # L-BFGS
    if cfg.use_lbfgs:
        lbfgs = optim.LBFGS(params, max_iter=cfg.lbfgs_max_iter, tolerance_change=1e-9, line_search_fn="strong_wolfe")
        lbfgs.step(closure)

    dt = time.time() - t0
    print(f"Entrenamiento terminado en {dt:.1f} s")

    # Guardado de modelo y parámetros
    ckpt = {
        'state_dict': model.state_dict(),
        'config': cfg.__dict__,
        'D': (
            (torch.exp(model.logD1).item(), torch.exp(model.logD2).item()) if cfg.use_bilayer else (torch.exp(model.logD).item(),)
        )
    }
    torch.save(ckpt, os.path.join(cfg.out_dir, f"ckpt_{cfg.out_tag}.pt"))

    # Exportar grilla de evaluación
    save_eval_grid(model, cfg, device)

    return model


# =========================================
# 9) Evaluación y exportación
# =========================================

def save_eval_grid(model: DiffusionPINN, cfg: PINNConfig, device: str = "cpu"):
    xg = np.linspace(0, cfg.L, cfg.grid_nx)
    tg = np.linspace(0, cfg.T, cfg.grid_nt)
    XX, TT = np.meshgrid(xg, tg)
    x_in = to_tensor(XX.reshape(-1, 1)).to(device)
    t_in = to_tensor(TT.reshape(-1, 1)).to(device)

    with torch.no_grad():
        C = model(x_in, t_in).cpu().numpy().reshape(cfg.grid_nt, cfg.grid_nx)

    out_csv = os.path.join(cfg.out_dir, f"grid_{cfg.out_tag}.csv")
    header = "x," + ",".join(f"{xi:.6e}" for xi in xg)
    np.savetxt(out_csv, np.column_stack([tg.reshape(-1, 1), C]), delimiter=",", header="t," + ",".join(map(str, xg)), comments="")
    print(f"Grilla evaluada guardada en: {out_csv}")


# =========================================
# 10) Ejecución por defecto
# =========================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Config base; ajusta según tu caso o lee desde args/JSON ----
    cfg = PINNConfig(
        L=1.0,
        T=1.0,
        use_bilayer=True,         # True para evaluar condición de interfaz
        x_if=0.4,
        D1_init=1.0e-10,
        D2_init=5.0e-11,
        D_const_init=8.0e-11,
        train_inverse=True,       # Estimar D1/D2 o D
        use_dirichlet_left=True,
        C_left_value=1.0,
        use_neumann_right=True,
        C0_value=0.0,
        N_f=6000,
        N_bc_t=1000,
        N_ic_x=1000,
        N_if_t=1000,
        N_data=0,                 # pon >0 si tendrás datos CSV
        w_pde=1.0,
        w_ic=2.0,
        w_bc=1.0,
        w_if=2.0,
        w_data=1.0,
        layers=(2, 128, 128, 128, 128, 1),
        act="tanh",
        lr=1e-3,
        adam_steps=15000,
        use_lbfgs=True,
        lbfgs_max_iter=3000,
        seed=42,
        grid_nx=201,
        grid_nt=101,
        out_dir="outputs_pinn",
        out_tag="pe_h2"
    )

    # Si tienes datos reales/sintéticos en CSV con columnas x,t,C, define su ruta aquí.
    data_csv_path = None  # por ejemplo: "data/h2_pe_measurements.csv"

    _ = train_pinn(cfg, data_csv=data_csv_path, device=device)
