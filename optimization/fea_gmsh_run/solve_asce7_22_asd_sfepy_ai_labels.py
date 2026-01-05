#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import numpy as nm

from sfepy.base.base import IndexedStruct, Struct
from sfepy.discrete import FieldVariable, Material, Integral, Equation, Equations, Problem
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.terms import Term
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.mechanics.tensors import get_von_mises_stress
from sfepy.discrete.common.region import Region


# ---------- regions ----------
def make_endplane_vertex_region_from_omega(domain, omega, name, axis=0, side="min"):
    coors = domain.get_mesh_coors()
    omega_vs = np.asarray(omega.vertices, dtype=np.int32)
    if omega_vs.size == 0:
        raise RuntimeError("Omega has 0 vertices - no 3D cells selected?")

    vals = coors[omega_vs, axis]
    vmin = float(vals.min())
    vmax = float(vals.max())
    extent = float(vmax - vmin)

    tol = 1e-6 * max(1.0, extent)
    for _ in range(10):
        if side == "min":
            thr = vmin + tol
            mask = vals <= thr
        else:
            thr = vmax - tol
            mask = vals >= thr

        sel = omega_vs[mask]
        if sel.size > 0:
            reg = Region.from_vertices(sel, domain, name=name, kind="vertex")
            dbg = dict(n=int(sel.size), thr=float(thr), tol=float(tol))
            return reg, dbg
        tol *= 10.0

    raise RuntimeError(f'Region "{name}" empty even after tolerance growth.')


def make_facet_region_box(domain, omega, name, axis, side):
    coors = domain.get_mesh_coors()
    omega_vs = np.asarray(omega.vertices, dtype=np.int32)
    vals = coors[omega_vs, axis]
    vmin = float(vals.min())
    vmax = float(vals.max())
    extent = float(vmax - vmin)

    tol = 1e-6 * max(1.0, extent)
    for _ in range(10):
        if side == "min":
            thr = vmin + tol
            cond = f"{'xyz'[axis]} < {thr}"
        else:
            thr = vmax - tol
            cond = f"{'xyz'[axis]} > {thr}"

        reg = domain.create_region(name, f"vertices in ({cond})", kind="facet", parent="Omega")
        if reg.facets is not None and len(reg.facets) > 0:
            return reg
        tol *= 10.0

    raise RuntimeError(f'Facet region "{name}" empty even after tolerance growth.')


# ---------- load materials ----------
def make_const_body_force_material(name, vec3):
    vec3 = np.asarray(vec3, dtype=float).reshape((1, 3, 1))

    def fun(ts, coors, mode=None, **kwargs):
        if mode == "qp":
            return {"val": np.tile(vec3, (coors.shape[0], 1, 1))}

    return Material(name, function=fun)


def make_const_pressure_material(name, p):
    p = float(p)

    def fun(ts, coors, mode=None, **kwargs):
        if mode == "qp":
            return {"val": np.tile(p, (coors.shape[0], 1, 1))}

    return Material(name, function=fun)


# ---------- postprocess ----------
def _u_nodal(variables):
    u_flat = np.asarray(variables.get_state_parts()["u"]).reshape((-1,))
    return u_flat.reshape((-1, 3))


def compute_ai_metrics(pb, variables, omega, m_solid, quad_order=2):
    out = variables.create_output()
    ev = pb.evaluate

    # Nodal displacement magnitude.
    u = _u_nodal(variables)
    u_mag = np.linalg.norm(u, axis=1)
    max_u = float(np.max(u_mag))

    # Cauchy stress, el_avg.
    stress = ev(
        f"ev_cauchy_stress.{quad_order}.{omega.name}(m.D, u)",
        mode="el_avg",
        copy_materials=False,
        m=m_solid,
    )
    out["cauchy_stress"] = Struct(name="output_data", mode="cell", data=stress, dofs=None)

    vms = get_von_mises_stress(stress.squeeze()).reshape((-1, 1, 1, 1))
    out["von_mises"] = Struct(name="output_data", mode="cell", data=vms, dofs=None)

    # Principal stresses from Voigt [sxx, syy, szz, syz, sxz, sxy].
    s = np.asarray(stress).reshape((stress.shape[0], -1))
    sxx, syy, szz, syz, sxz, sxy = (s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5])
    T = np.zeros((s.shape[0], 3, 3), dtype=float)
    T[:, 0, 0] = sxx
    T[:, 1, 1] = syy
    T[:, 2, 2] = szz
    T[:, 1, 2] = T[:, 2, 1] = syz
    T[:, 0, 2] = T[:, 2, 0] = sxz
    T[:, 0, 1] = T[:, 1, 0] = sxy
    w = np.linalg.eigvalsh(T)  # ascending
    s3, s2, s1 = w[:, 0], w[:, 1], w[:, 2]

    out["sigma_1"] = Struct(name="output_data", mode="cell", data=s1.reshape((-1, 1, 1, 1)), dofs=None)
    out["sigma_3"] = Struct(name="output_data", mode="cell", data=s3.reshape((-1, 1, 1, 1)), dofs=None)

    # Volume of Omega; SfePy supports evaluators like ev_volume.* [web:304]
    vol = float(ev(f"ev_volume.{quad_order}.{omega.name}(u)", mode="eval"))

    # "Compliance-like" integral: ∫ e(u):D:e(u) dΩ ; strain energy = 0.5*that. [web:1]
    a_uu = float(
        ev(
            f"dw_lin_elastic.{quad_order}.{omega.name}(m.D, u, u)",
            mode="eval",
            copy_materials=False,
            m=m_solid,
        )
    )
    strain_energy = 0.5 * a_uu
    compliance_like = a_uu

    met = {
        "volume_m3": vol,
        "max_u": max_u,
        "max_sigma_1": float(np.max(s1)),
        "min_sigma_3": float(np.min(s3)),
        "max_abs_sigma_3_comp": float(np.max(np.abs(np.minimum(s3, 0.0)))),
        "max_von_mises": float(np.max(vms)),
        "strain_energy": float(strain_energy),
        "compliance_like": float(compliance_like),
    }
    return out, met


# ---------- ASD combos ----------
def asd_combos_practical():
    # Practical ASD set (verify against your licensed ASCE 7-22). [web:22]
    return [
        ("D", {"D": 1.0}),
        ("D+L", {"D": 1.0, "L": 1.0}),
        ("D+0.6W", {"D": 1.0, "W": 0.6}),
        ("D+0.7E", {"D": 1.0, "E": 0.7}),
        ("0.6D+0.6W", {"D": 0.6, "W": 0.6}),
        ("0.6D+0.7E", {"D": 0.6, "E": 0.7}),
        ("D+0.75L+0.45W", {"D": 1.0, "L": 0.75, "W": 0.45}),
        ("D+0.75L+0.525E", {"D": 1.0, "L": 0.75, "E": 0.525}),
    ]


# ---------- Load spec helpers ----------
def load_body(tag, vec_force_density):  # N/m^3
    return {"type": "body", "tag": tag, "f": np.asarray(vec_force_density, dtype=float)}


def load_press(tag, region, pressure):  # Pa
    return {"type": "press", "tag": tag, "region": region, "p": float(pressure)}


def loads_for_symbol(sym, fac, *, rho, g, live_az):
    if abs(fac) < 1e-15:
        return []

    if sym == "D":
        # body force density = rho * gvec
        return [load_body("fD", fac * rho * np.array([0.0, 0.0, -abs(g)], dtype=float))]

    if sym == "L":
        return [load_body("fL", fac * rho * np.array([0.0, 0.0, -float(live_az)], dtype=float))]

    raise ValueError(sym)


def loads_for_E_direction(fac, direction, *, rho, ex, ey):
    if abs(fac) < 1e-15:
        return []

    if direction == "Ex+":
        ax, ay = float(ex), 0.0
    elif direction == "Ex-":
        ax, ay = -float(ex), 0.0
    elif direction == "Ey+":
        ax, ay = 0.0, float(ey)
    elif direction == "Ey-":
        ax, ay = 0.0, -float(ey)
    else:
        raise ValueError(direction)

    # inertia opposite accel.
    f = rho * np.array([-ax, -ay, 0.0], dtype=float)
    return [load_body("fE", fac * f)]


def loads_for_W_direction(fac, direction, *, x_plus, x_minus, y_plus, y_minus, px, nx, py, ny):
    if abs(fac) < 1e-15:
        return []

    if direction == "Wx+":
        return [load_press("pW", x_plus, fac * float(px))]
    if direction == "Wx-":
        return [load_press("pW", x_minus, fac * float(nx))]
    if direction == "Wy+":
        return [load_press("pW", y_plus, fac * float(py))]
    if direction == "Wy-":
        return [load_press("pW", y_minus, fac * float(ny))]

    raise ValueError(direction)


# ---------- solve one RHS (loads are dicts, not Terms) ----------
def solve_with_loads(
    tag,
    domain,
    omega,
    left_v,
    young, poisson,
    quad,
    loads,
    write_vtk_path: Path | None,
):
    field = Field.from_args("fu", nm.float64, "vector", omega, approx_order=1)
    u = FieldVariable("u", "unknown", field)
    v = FieldVariable("v", "test", field, primary_var_name="u")

    Dmat = stiffness_from_youngpoisson(3, young=young, poisson=poisson)
    m_solid = Material("m", D=Dmat)

    integral = Integral("i", order=int(quad))

    t_el = Term.new("dw_lin_elastic(m.D, v, u)", integral, omega, m=m_solid, v=v, u=u)  # [web:1]

    rhs_terms = []
    for ld in loads:
        if ld["type"] == "body":
            mat = make_const_body_force_material(ld["tag"], ld["f"])
            rhs_terms.append(
                Term.new(f"dw_volume_lvf({ld['tag']}.val, v)", integral, omega, **{ld["tag"]: mat}, v=v)  # [web:58]
            )
        elif ld["type"] == "press":
            mat = make_const_pressure_material(ld["tag"], ld["p"])
            rhs_terms.append(
                Term.new(
                    f"dw_surface_ltr({ld['tag']}.val, v)",
                    integral,
                    ld["region"],
                    **{ld["tag"]: mat},
                    v=v,
                )
            )
        else:
            raise ValueError(f"Unknown load type: {ld['type']}")

    if not rhs_terms:
        raise RuntimeError(f"{tag}: no RHS terms! (all loads were zero?)")

    rhs = rhs_terms[0]
    for t in rhs_terms[1:]:
        rhs = rhs + t

    eq = Equation("balance", t_el + rhs)
    eqs = Equations([eq])

    bc_fix = EssentialBC("bc_fix", left_v, {"u.all": 0.0})

    ls = ScipyDirect({})
    nls = Newton({}, lin_solver=ls, status=IndexedStruct())
    pb = Problem(f"solve_{tag}", equations=eqs)
    pb.set_bcs(ebcs=Conditions([bc_fix]))
    pb.set_solver(nls)

    variables = pb.solve(status=IndexedStruct())
    out, met = compute_ai_metrics(pb, variables, omega, m_solid, quad_order=int(quad))

    if write_vtk_path is not None:
        write_vtk_path.parent.mkdir(parents=True, exist_ok=True)
        pb.save_state(str(write_vtk_path), out=out)

    return met


def solve_named_case(
    case_name,
    domain, omega, left_v,
    x_plus, x_minus, y_plus, y_minus,
    args,
    out_vtk: Path | None,
):
    if case_name == "D":
        loads = loads_for_symbol("D", 1.0, rho=args.rho, g=args.g, live_az=args.live_az)
    elif case_name == "L":
        loads = loads_for_symbol("L", 1.0, rho=args.rho, g=args.g, live_az=args.live_az)
    elif case_name in ("Ex+", "Ex-", "Ey+", "Ey-"):
        loads = loads_for_E_direction(1.0, case_name, rho=args.rho, ex=args.ex, ey=args.ey)
    elif case_name in ("Wx+", "Wx-", "Wy+", "Wy-"):
        loads = loads_for_W_direction(
            1.0, case_name,
            x_plus=x_plus, x_minus=x_minus, y_plus=y_plus, y_minus=y_minus,
            px=args.px, nx=args.nx, py=args.py, ny=args.ny
        )
    else:
        raise ValueError(case_name)

    return solve_with_loads(
        f"case_{case_name}",
        domain, omega, left_v,
        args.young, args.poisson,
        args.quad,
        loads,
        out_vtk,
    )


def solve_combo_true_envelope(
    combo_name,
    combo_dict,
    domain, omega, left_v,
    x_plus, x_minus, y_plus, y_minus,
    args,
    out_dir: Path,
    write_combo_vtk: bool,
):
    # Base loads (D/L parts).
    base = []
    for sym, fac in combo_dict.items():
        if sym in ("D", "L"):
            base += loads_for_symbol(sym, fac, rho=args.rho, g=args.g, live_az=args.live_az)

    need_W = ("W" in combo_dict) and (abs(combo_dict["W"]) > 0)
    need_E = ("E" in combo_dict) and (abs(combo_dict["E"]) > 0)

    scenarios = [("only", [])]

    if need_W:
        facW = combo_dict["W"]
        scenarios = [
            (f"{tagW}", extraW)
            for (_old_tag, old_extra) in scenarios
            for (tagW, extraW) in [
                ("Wx+", old_extra + loads_for_W_direction(facW, "Wx+", x_plus=x_plus, x_minus=x_minus, y_plus=y_plus, y_minus=y_minus,
                                                         px=args.px, nx=args.nx, py=args.py, ny=args.ny)),
                ("Wx-", old_extra + loads_for_W_direction(facW, "Wx-", x_plus=x_plus, x_minus=x_minus, y_plus=y_plus, y_minus=y_minus,
                                                         px=args.px, nx=args.nx, py=args.py, ny=args.ny)),
                ("Wy+", old_extra + loads_for_W_direction(facW, "Wy+", x_plus=x_plus, x_minus=x_minus, y_plus=y_plus, y_minus=y_minus,
                                                         px=args.px, nx=args.nx, py=args.py, ny=args.ny)),
                ("Wy-", old_extra + loads_for_W_direction(facW, "Wy-", x_plus=x_plus, x_minus=x_minus, y_plus=y_plus, y_minus=y_minus,
                                                         px=args.px, nx=args.nx, py=args.py, ny=args.ny)),
            ]
        ]

    if need_E:
        facE = combo_dict["E"]
        scenarios = [
            (f"{tagE}", extraE)
            for (_old_tag, old_extra) in scenarios
            for (tagE, extraE) in [
                ("Ex+", old_extra + loads_for_E_direction(facE, "Ex+", rho=args.rho, ex=args.ex, ey=args.ey)),
                ("Ex-", old_extra + loads_for_E_direction(facE, "Ex-", rho=args.rho, ex=args.ex, ey=args.ey)),
                ("Ey+", old_extra + loads_for_E_direction(facE, "Ey+", rho=args.rho, ex=args.ex, ey=args.ey)),
                ("Ey-", old_extra + loads_for_E_direction(facE, "Ey-", rho=args.rho, ex=args.ex, ey=args.ey)),
            ]
        ]

    # If neither W nor E exist, solve once.
    if (not need_W) and (not need_E):
        scenarios = [("only", [])]

    keys_env_max = ["max_u", "max_sigma_1", "max_abs_sigma_3_comp", "max_von_mises", "strain_energy", "compliance_like"]
    keys_env_min = ["min_sigma_3"]

    best = None
    for scen_tag, scen_extra in scenarios:
        loads = base + scen_extra

        vtk_path = None
        if write_combo_vtk:
            safe = "".join(ch if ch.isalnum() or ch in "+-_" else "_" for ch in combo_name)
            vtk_path = out_dir / f"combo_{safe}_{scen_tag}.vtk"

        met = solve_with_loads(
            f"combo_{combo_name}_{scen_tag}",
            domain, omega, left_v,
            args.young, args.poisson,
            args.quad,
            loads,
            vtk_path,
        )

        if best is None:
            best = met
            best["scenario_envelope"] = scen_tag
        else:
            for k in keys_env_max:
                best[k] = max(best[k], met[k])
            for k in keys_env_min:
                best[k] = min(best[k], met[k])

    return best


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("msh", type=str)
    ap.add_argument("--out-dir", type=str, default="fea_out_asce7_22_asd_ai")
    ap.add_argument("--quad", type=int, default=2)

    ap.add_argument("--young", type=float, default=25e9)
    ap.add_argument("--poisson", type=float, default=0.20)

    ap.add_argument("--rho", type=float, default=2400.0)
    ap.add_argument("--g", type=float, default=9.81)

    ap.add_argument("--live-az", type=float, default=0.0)

    ap.add_argument("--ex", type=float, default=0.0)
    ap.add_argument("--ey", type=float, default=0.0)

    ap.add_argument("--px", type=float, default=0.0)
    ap.add_argument("--nx", type=float, default=0.0)
    ap.add_argument("--py", type=float, default=0.0)
    ap.add_argument("--ny", type=float, default=0.0)

    ap.add_argument("--write-case-vtk", action="store_true")
    ap.add_argument("--write-combo-vtk", action="store_true")
    args = ap.parse_args()

    msh_path = Path(args.msh)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = Mesh.from_file(str(msh_path))
    domain = FEDomain("domain", mesh)
    omega = domain.create_region("Omega", "all", extra_options={"cell_tdim": 3}, allow_empty=False)

    left_v, dbg = make_endplane_vertex_region_from_omega(domain, omega, "LeftV", axis=0, side="min")
    print(f"LeftV: n={dbg['n']} thr={dbg['thr']:.6g} tol={dbg['tol']:.3e}")

    x_plus = make_facet_region_box(domain, omega, "XPlus", axis=0, side="max")
    x_minus = make_facet_region_box(domain, omega, "XMinus", axis=0, side="min")
    y_plus = make_facet_region_box(domain, omega, "YPlus", axis=1, side="max")
    y_minus = make_facet_region_box(domain, omega, "YMinus", axis=1, side="min")

    # ----- Base cases (debug/optional) -----
    cases = ["D"]
    if abs(args.live_az) > 0:
        cases.append("L")
    if abs(args.ex) > 0:
        cases += ["Ex+", "Ex-"]
    if abs(args.ey) > 0:
        cases += ["Ey+", "Ey-"]
    if args.px != 0:
        cases.append("Wx+")
    if args.nx != 0:
        cases.append("Wx-")
    if args.py != 0:
        cases.append("Wy+")
    if args.ny != 0:
        cases.append("Wy-")

    case_rows = []
    for c in cases:
        vtk_path = (out_dir / f"{msh_path.stem}_case_{c}.vtk") if args.write_case_vtk else None
        met = solve_named_case(c, domain, omega, left_v, x_plus, x_minus, y_plus, y_minus, args, vtk_path)
        met["mesh"] = msh_path.name
        met["type"] = "case"
        met["name"] = c
        met["mass_kg"] = met["volume_m3"] * float(args.rho)
        case_rows.append(met)
        print(f"[case {c}] max_u={met['max_u']:.6g}  max_sigma1={met['max_sigma_1']:.6g} Pa  max_comp={met['max_abs_sigma_3_comp']:.6g} Pa  energy={met['strain_energy']:.6g}")

    # ----- True combo solves (AI labels) -----
    combo_rows = []
    print("\n=== TRUE ASD combo envelopes (solved) ===")
    for name, combo in asd_combos_practical():
        met = solve_combo_true_envelope(
            name, combo,
            domain, omega, left_v,
            x_plus, x_minus, y_plus, y_minus,
            args,
            out_dir=out_dir,
            write_combo_vtk=bool(args.write_combo_vtk),
        )
        met["mesh"] = msh_path.name
        met["type"] = "combo"
        met["name"] = name
        met["mass_kg"] = met["volume_m3"] * float(args.rho)
        combo_rows.append(met)
        print(f"{name:16s}  max_u={met['max_u']:.6g}  max_sigma1={met['max_sigma_1']:.6g} Pa  max_comp={met['max_abs_sigma_3_comp']:.6g} Pa  energy={met['strain_energy']:.6g}")

    # ----- Write CSV -----
    fields = [
        "mesh", "type", "name",
        "volume_m3", "mass_kg",
        "max_u",
        "max_sigma_1", "min_sigma_3", "max_abs_sigma_3_comp",
        "max_von_mises",
        "strain_energy", "compliance_like",
        "scenario_envelope",
    ]
    write_csv(out_dir / "fea_labels_cases.csv", case_rows, fields)
    write_csv(out_dir / "fea_labels_combos.csv", combo_rows, fields)

    print(f"\nWrote: {out_dir / 'fea_labels_cases.csv'}")
    print(f"Wrote: {out_dir / 'fea_labels_combos.csv'}")
    print("\nNOTE: Wind/seismic magnitudes must be computed per ASCE 7-22 for your site/building. [web:22]")


if __name__ == "__main__":
    main()
