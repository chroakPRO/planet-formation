#!/usr/bin/env python3
"""
Multi-system N-body planet formation with REBOUND, auto-sized to ~target wall time.

What this does
--------------
- Runs several independent star systems (each its own N-body sim).
- Seeds embryos + planetesimals in a disk (A_MIN..A_MAX).
- Collisions: perfect merges (inelastic).
- Colors bodies by where they STARTED relative to the snow line (cyan=icy/water-rich, orange=dry).
- Optionally makes an MP4 per system (set ANIMATE=False to save time).
- Auto-sizes the simulated duration (and if needed, planetesimal count) to fit your time budget,
  using a quick micro-benchmark on your machine.

IMPORTANT
---------
- This is still computationally heavy. If it's too slow, reduce N_PLANETESIMALS, N_SYSTEMS, or set ANIMATE=False.
- Water tracking is origin-based only (no mixing during mergers). Ask if you want a proper collision callback
  that propagates a water_fraction through mergers.

Dependencies:
    pip install rebound matplotlib pandas
    # for MP4:
    # macOS: brew install ffmpeg
    # Ubuntu/Debian: sudo apt install ffmpeg
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import rebound
from pathlib import Path
import os

# Ensure ffmpeg is in PATH
os.environ['PATH'] = '/opt/homebrew/bin:/usr/local/bin:' + os.environ.get('PATH', '')

# -----------------------
# USER CONFIG (edit these)
# -----------------------
TARGET_MINUTES = 60            # aim for ~this many minutes in total (1 hour)
N_SYSTEMS = 2                  # number of stars/systems to simulate
ANIMATE = True                 # save MP4 per system (False = faster)
MAX_FRAMES_PER_SYSTEM = 120    # cap frames so animation doesn't dominate runtime

# Disk/physics layout (friendly defaults for stability + speed)
A_MIN = 0.8                    # AU (inner edge; farther out allows larger dt)
A_MAX = 4.0                    # AU (outer edge)
M_STAR_MIN = 0.7               # Msun
M_STAR_MAX = 1.1               # Msun

# Initial body counts (auto-sizer may reduce planetesimals if needed)
N_EMBRYOS = 40
N_PLANETESIMALS = 100

# Timestep: for A_MIN=0.8 AU, P_inner ~ 0.72 yr -> dt ~ P/30 ≈ 0.024 yr; we go a bit smaller for safety.
DT_YEARS = 0.02                # ~7.3 days; smaller = more accurate, slower

# Initial sim duration guess (auto-sizer will scale this)  
SIM_YEARS_GUESS = 1e6          # 1 Myr per system (meaningful planet formation)
FRAME_INTERVAL_YEARS = 5e3     # 5,000 years per frame (200 frames for full sim)
OUTPUT_INTERVAL_YEARS = 0.5e6  # progress print interval

# Habitability heuristics
HZ_IN_SUN = 0.95               # AU (for Sun)
HZ_OUT_SUN = 1.67              # AU
ROCKY_MIN = 0.5                # Earth masses
ROCKY_MAX = 5.0                # Earth masses

# -----------------------
# Helpers
# -----------------------
def star_luminosity(mstar):
    """Very rough main-sequence scaling: L ~ M^3.5 (in solar units)."""
    return mstar**3.5

def habitable_zone(lum):
    """Conservative Solar-scaled HZ bounds (in AU)."""
    return HZ_IN_SUN*sqrt(lum), HZ_OUT_SUN*sqrt(lum)

def snow_line(lum):
    """Iceline/snow line scaling (in AU)."""
    return 2.7*sqrt(lum)

def build_sim(n_planetesimals, n_embryos, a_min, a_max, dt, m_star, rng):
    """Create a REBOUND sim: star + embryos + planetesimals, low e/i, perfect-merge collisions."""
    sim = rebound.Simulation()
    sim.units = ("AU","yr","Msun")
    sim.add(m=m_star)

    # Embryos (bigger seeds)
    for _ in range(n_embryos):
        a = rng.uniform(a_min, a_max)
        e = rng.uniform(0, 0.02)
        inc = rng.uniform(0, 0.5) * np.pi/180
        mass = rng.uniform(0.01, 0.05) * 3e-6  # Mearth->Msun (1 Mearth = 3e-6 Msun)
        sim.add(m=mass, a=a, e=e, inc=inc)

    # Planetesimals (smaller seeds)
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

def micro_benchmark(n_planetesimals, n_embryos, dt, a_min, a_max):
    """
    Build a small sim and time a short integration to infer time per step on this machine.
    Returns time_per_step (seconds).
    """
    rng = np.random.default_rng(123)
    m_star = 1.0
    sim = build_sim(n_planetesimals, n_embryos, a_min, a_max, dt, m_star, rng)

    # Warm-up (stabilize caches)
    warm_years = 20.0
    sim.integrate(sim.t + warm_years)

    bench_years = 100.0  # short but enough steps to time
    steps = int(round(bench_years / dt))
    t0 = time.perf_counter()
    sim.integrate(sim.t + bench_years)
    elapsed = time.perf_counter() - t0
    time_per_step = elapsed / max(steps, 1)
    return time_per_step

def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

# -----------------------
# Auto-size to target time
# -----------------------
def autosize_to_budget():
    """
    Measure time/step, estimate total time, then:
      1) Scale simulated years to fit target,
      2) If still too slow, reduce planetesimals progressively,
      3) Cap animation frames.
    Returns (sim_years, n_planetesimals, frame_interval, est_total_seconds).
    """
    target_seconds = TARGET_MINUTES * 60.0
    print("Benchmarking your machine...")
    tps = micro_benchmark(N_PLANETESIMALS, N_EMBRYOS, DT_YEARS, A_MIN, A_MAX)
    print(f"Measured time/step ≈ {tps*1e3:.3f} ms (with N≈{N_PLANETESIMALS+N_EMBRYOS+1})")

    # Estimate steps per system for guess duration
    steps_guess = int(round(SIM_YEARS_GUESS / DT_YEARS))
    # Collisions reduce N over time; apply a mild 0.7 factor to total steps cost
    fudge = 0.7
    est_per_system = tps * steps_guess * fudge
    est_total = est_per_system * N_SYSTEMS
    print(f"Initial estimate for SIM_YEARS={SIM_YEARS_GUESS:.2e}: ~{est_total/60:.1f} minutes total")

    sim_years = SIM_YEARS_GUESS
    n_plan = N_PLANETESIMALS

    # If too slow, first scale sim years down to fit target
    if est_total > target_seconds:
        scale = target_seconds / est_total
        sim_years = max(1e6, SIM_YEARS_GUESS * scale)  # don't go below 1 Myr
        steps = int(round(sim_years / DT_YEARS))
        est_per_system = tps * steps * fudge
        est_total = est_per_system * N_SYSTEMS
        print(f"Scaled SIM_YEARS -> {sim_years:.2e}; new estimate ~{est_total/60:.1f} minutes")

    # If still too slow, reduce planetesimals (tps depends on N; re-benchmark each reduction)
    while est_total > target_seconds and n_plan > 50:
        n_plan = int(n_plan * 0.8)
        tps = micro_benchmark(n_plan, N_EMBRYOS, DT_YEARS, A_MIN, A_MAX)
        steps = int(round(sim_years / DT_YEARS))
        est_per_system = tps * steps * fudge
        est_total = est_per_system * N_SYSTEMS
        print(f"Reduced N_PLANETESIMALS -> {n_plan}; estimate ~{est_total/60:.1f} minutes")

    # Cap frames to avoid animation overhead
    frames_cap = MAX_FRAMES_PER_SYSTEM
    frame_interval = max(sim_years / frames_cap, FRAME_INTERVAL_YEARS)

    return sim_years, n_plan, frame_interval, est_total

# -----------------------
# Main run
# -----------------------
def main():
    sim_years, n_planetesimals, frame_interval, est_total = autosize_to_budget()
    print(f"\n=== Final plan ===")
    print(f"Systems: {N_SYSTEMS} | Embryos: {N_EMBRYOS} | Planetesimals: {n_planetesimals}")
    print(f"dt: {DT_YEARS} yr | SIM_YEARS per system: {sim_years:.2e} | Frame Δt: {frame_interval:.2e} yr")
    print(f"Estimated total wall-time: ~{est_total/60:.1f} minutes (hardware dependent)")

    rng = np.random.default_rng(42)
    all_rows = []

    for sys_id in range(1, N_SYSTEMS+1):
        print(f"\n=== System {sys_id}/{N_SYSTEMS} ===")
        m_star = rng.uniform(M_STAR_MIN, M_STAR_MAX)
        lum = star_luminosity(m_star)
        hz_in, hz_out = habitable_zone(lum)
        r_snow = snow_line(lum)

        sim = build_sim(n_planetesimals, N_EMBRYOS, A_MIN, A_MAX, DT_YEARS, m_star, rng)

        # Tag water by origin (no mixing through collisions in this version)
        origins = []
        for p in sim.particles[1:]:
            origins.append(p.a >= r_snow)

        if ANIMATE:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=15, metadata=dict(artist='REBOUND'), bitrate=1600)
            fig, ax = plt.subplots(figsize=(6,6))
            anim_file = f"system_{sys_id}_formation.mp4"
            ensure_dir(anim_file)
            with writer.saving(fig, anim_file, dpi=140):
                t = 0.0
                while t < sim_years:
                    sim.integrate(t)
                    ax.clear()
                    ax.set_facecolor("black")
                    ax.set_xlim(-6, 6)
                    ax.set_ylim(-6, 6)
                    ax.set_xlabel("x [AU]")
                    ax.set_ylabel("y [AU]")
                    ax.set_title(f"System {sys_id}  t={t/1e6:.1f} Myr  N={sim.N}")

                    xs, ys, ss, cs = [], [], [], []
                    # Skip star (index 0)
                    for i, p in enumerate(sim.particles[1:]):
                        xs.append(p.x); ys.append(p.y)
                        m_me = p.m / 3e-6
                        ss.append(np.clip(np.sqrt(m_me)*2, 1, 18))
                        cs.append("cyan" if origins[i] else "orange")

                    ax.scatter(xs, ys, s=ss, c=cs, alpha=0.85)
                    # HZ rings
                    for r in (hz_in, hz_out):
                        circ = plt.Circle((0,0), r, color="yellow", fill=False, alpha=0.35)
                        ax.add_artist(circ)

                    writer.grab_frame()
                    t += frame_interval

                plt.close(fig)
        else:
            # Non-animated integration with periodic reporting
            t, next_report = 0.0, OUTPUT_INTERVAL_YEARS
            while t < sim_years:
                sim.integrate(min(t + OUTPUT_INTERVAL_YEARS, sim_years))
                t = sim.t
                if t >= next_report:
                    print(f"  t={t/1e6:.1f} Myr  N={sim.N}")
                    next_report += OUTPUT_INTERVAL_YEARS

        # Collect final survivors
        for i, p in enumerate(sim.particles[1:]):
            m_me = p.m / 3e-6
            in_hz = (p.a >= hz_in) and (p.a <= hz_out)
            rocky = (ROCKY_MIN <= m_me <= ROCKY_MAX)
            water_rich = origins[i]  # origin-based heuristic
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
    df.to_csv("autosized_final_planets.csv", index=False)

    # Summary per system
    summary = df.groupby("system_id").agg(
        planets=("a_AU", "count"),
        hz_rocky=("life_capable", "sum")
    )
    print("\n=== Summary ===")
    print(summary)
    print("\nOutputs: autosized_final_planets.csv and per-system MP4s (if ANIMATE=True).")

if __name__ == "__main__":
    main()