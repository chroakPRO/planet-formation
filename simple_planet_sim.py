#!/usr/bin/env python3
"""
Simple planet formation simulation with static plots (no MP4 dependency)
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import rebound

# Quick demo config
TARGET_MINUTES = 0.5
N_SYSTEMS = 2
N_EMBRYOS = 10
N_PLANETESIMALS = 20
A_MIN = 0.8
A_MAX = 4.0
M_STAR_MIN = 0.7
M_STAR_MAX = 1.1
DT_YEARS = 0.02
SIM_YEARS = 1e4  # 10,000 years

def star_luminosity(mstar):
    return mstar**3.5

def habitable_zone(lum):
    return 0.95*sqrt(lum), 1.67*sqrt(lum)

def snow_line(lum):
    return 2.7*sqrt(lum)

def build_sim(n_planetesimals, n_embryos, a_min, a_max, dt, m_star, rng):
    sim = rebound.Simulation()
    sim.units = ("AU","yr","Msun")
    sim.add(m=m_star)

    # Embryos
    for _ in range(n_embryos):
        a = rng.uniform(a_min, a_max)
        e = rng.uniform(0, 0.02)
        inc = rng.uniform(0, 0.5) * np.pi/180
        mass = rng.uniform(0.01, 0.05) * 3e-6
        sim.add(m=mass, a=a, e=e, inc=inc)

    # Planetesimals
    for _ in range(n_planetesimals):
        a = rng.uniform(a_min, a_max)
        e = rng.uniform(0, 0.02)
        inc = rng.uniform(0, 0.5) * np.pi/180
        mass = rng.uniform(0.0005, 0.002) * 3e-6
        sim.add(m=mass, a=a, e=e, inc=inc)

    sim.move_to_com()
    sim.integrator = "whfast"
    sim.dt = dt
    sim.collision = "direct"
    sim.collision_resolve = "merge"
    return sim

def main():
    print(f"Running quick planet formation demo...")
    print(f"Systems: {N_SYSTEMS} | Embryos: {N_EMBRYOS} | Planetesimals: {N_PLANETESIMALS}")
    print(f"Simulation time: {SIM_YEARS:.0f} years per system")
    
    rng = np.random.default_rng(42)
    all_rows = []

    for sys_id in range(1, N_SYSTEMS+1):
        print(f"\n=== System {sys_id}/{N_SYSTEMS} ===")
        m_star = rng.uniform(M_STAR_MIN, M_STAR_MAX)
        lum = star_luminosity(m_star)
        hz_in, hz_out = habitable_zone(lum)
        r_snow = snow_line(lum)

        sim = build_sim(N_PLANETESIMALS, N_EMBRYOS, A_MIN, A_MAX, DT_YEARS, m_star, rng)

        # Tag water by origin
        origins = []
        for p in sim.particles[1:]:
            origins.append(p.a >= r_snow)

        # Create static plot before integration
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Initial state
        ax1.set_facecolor("black")
        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-5, 5)
        ax1.set_title(f"System {sys_id} - Initial State")
        ax1.set_xlabel("x [AU]")
        ax1.set_ylabel("y [AU]")
        
        xs, ys, ss, cs = [], [], [], []
        for i, p in enumerate(sim.particles[1:]):
            xs.append(p.x); ys.append(p.y)
            m_me = p.m / 3e-6
            ss.append(np.clip(np.sqrt(m_me)*3, 2, 15))
            cs.append("cyan" if origins[i] else "orange")
        
        ax1.scatter(xs, ys, s=ss, c=cs, alpha=0.8)
        for r in (hz_in, hz_out):
            circ = plt.Circle((0,0), r, color="yellow", fill=False, alpha=0.4, linewidth=2)
            ax1.add_artist(circ)

        # Run simulation
        print(f"  Integrating {SIM_YEARS:.0f} years...")
        sim.integrate(SIM_YEARS)
        print(f"  Final body count: {sim.N}")

        # Final state
        ax2.set_facecolor("black")
        ax2.set_xlim(-5, 5)
        ax2.set_ylim(-5, 5)
        ax2.set_title(f"System {sys_id} - After {SIM_YEARS:.0f} years")
        ax2.set_xlabel("x [AU]")
        ax2.set_ylabel("y [AU]")
        
        xs, ys, ss, cs = [], [], [], []
        for i, p in enumerate(sim.particles[1:]):
            xs.append(p.x); ys.append(p.y)
            m_me = p.m / 3e-6
            ss.append(np.clip(np.sqrt(m_me)*3, 2, 20))
            cs.append("cyan" if origins[i] else "orange")
        
        ax2.scatter(xs, ys, s=ss, c=cs, alpha=0.8)
        for r in (hz_in, hz_out):
            circ = plt.Circle((0,0), r, color="yellow", fill=False, alpha=0.4, linewidth=2)
            ax2.add_artist(circ)

        plt.tight_layout()
        plt.savefig(f"system_{sys_id}_formation.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Collect final data
        for i, p in enumerate(sim.particles[1:]):
            m_me = p.m / 3e-6
            in_hz = (p.a >= hz_in) and (p.a <= hz_out)
            rocky = (0.5 <= m_me <= 5.0)
            water_rich = origins[i]
            life_capable = in_hz and rocky and water_rich
            all_rows.append({
                "system_id": sys_id,
                "star_mass_Msun": m_star,
                "luminosity_Lsun": lum,
                "a_AU": p.a,
                "mass_Mearth": m_me,
                "e": p.e,
                "inc_deg": p.inc*180/np.pi,
                "in_HZ": in_hz,
                "water_rich_origin": water_rich,
                "life_capable": life_capable
            })

    df = pd.DataFrame(all_rows)
    df.to_csv("planet_formation_results.csv", index=False)

    # Summary
    summary = df.groupby("system_id").agg(
        planets=("a_AU", "count"),
        hz_planets=("in_HZ", "sum"),
        life_capable=("life_capable", "sum")
    )
    print("\n=== Summary ===")
    print(summary)
    print(f"\nOutputs: planet_formation_results.csv + system_X_formation.png plots")

if __name__ == "__main__":
    main()