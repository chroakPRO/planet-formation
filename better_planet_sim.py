#!/usr/bin/env python3
"""
Better planet formation simulation with smooth animation and meaningful dynamics
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from math import sqrt
import rebound
from pathlib import Path
import os

# Ensure ffmpeg is in PATH
os.environ['PATH'] = '/opt/homebrew/bin:/usr/local/bin:' + os.environ.get('PATH', '')

# Config for meaningful planet formation
N_SYSTEMS = 1                   # Focus on one system for better detail
N_EMBRYOS = 15                  # Fewer but more massive starting bodies
N_PLANETESIMALS = 50            # More collision targets
A_MIN = 0.5                     # Inner edge  
A_MAX = 3.0                     # Outer edge
M_STAR = 1.0                    # Solar mass
DT_YEARS = 0.01                 # Smaller timestep for stability
SIM_YEARS = 1e5                 # 100,000 years (shorter but more active)
FRAME_INTERVAL_YEARS = 500      # 500 years per frame = smooth motion
ANIMATION_FPS = 20              # Smooth video

def star_luminosity(mstar):
    return mstar**3.5

def habitable_zone(lum):
    return 0.95*sqrt(lum), 1.67*sqrt(lum)

def snow_line(lum):
    return 2.7*sqrt(lum)

def build_sim(n_planetesimals, n_embryos, a_min, a_max, dt, m_star, rng):
    sim = rebound.Simulation()
    sim.units = ("AU","yr","Msun")
    sim.add(m=m_star)  # Central star

    # Embryos - larger starting masses for more dramatic collisions
    for _ in range(n_embryos):
        a = rng.uniform(a_min, a_max)
        # Give them different eccentricities for crossing orbits
        e = rng.uniform(0.01, 0.1)  # Higher eccentricity = more collisions
        inc = rng.uniform(0, 2.0) * np.pi/180  # Small inclinations
        mass = rng.uniform(0.1, 0.5) * 3e-6   # Bigger embryos (0.1-0.5 Earth masses)
        # Add some orbital velocity spread
        Omega = rng.uniform(0, 2*np.pi)
        omega = rng.uniform(0, 2*np.pi) 
        f = rng.uniform(0, 2*np.pi)
        sim.add(m=mass, a=a, e=e, inc=inc, Omega=Omega, omega=omega, f=f)

    # Planetesimals - smaller but numerous
    for _ in range(n_planetesimals):
        a = rng.uniform(a_min, a_max)
        e = rng.uniform(0.005, 0.05)  # Some eccentricity
        inc = rng.uniform(0, 1.0) * np.pi/180
        mass = rng.uniform(0.005, 0.02) * 3e-6  # Smaller planetesimals
        Omega = rng.uniform(0, 2*np.pi)
        omega = rng.uniform(0, 2*np.pi)
        f = rng.uniform(0, 2*np.pi)
        sim.add(m=mass, a=a, e=e, inc=inc, Omega=Omega, omega=omega, f=f)

    sim.move_to_com()
    sim.integrator = "whfast"
    sim.dt = dt
    sim.collision = "direct"
    sim.collision_resolve = "merge"
    
    return sim

def main():
    print(f"Running enhanced planet formation simulation...")
    print(f"Simulation time: {SIM_YEARS:.0f} years")
    print(f"Frame interval: {FRAME_INTERVAL_YEARS:.0f} years")
    print(f"Expected frames: {int(SIM_YEARS/FRAME_INTERVAL_YEARS)}")
    print(f"Video length: ~{int(SIM_YEARS/FRAME_INTERVAL_YEARS)/ANIMATION_FPS:.1f} seconds")
    
    rng = np.random.default_rng(42)
    m_star = M_STAR
    lum = star_luminosity(m_star)
    hz_in, hz_out = habitable_zone(lum)
    r_snow = snow_line(lum)

    print(f"\nStar: {m_star:.1f} solar masses")
    print(f"Habitable zone: {hz_in:.2f} - {hz_out:.2f} AU")
    print(f"Snow line: {r_snow:.2f} AU")

    sim = build_sim(N_PLANETESIMALS, N_EMBRYOS, A_MIN, A_MAX, DT_YEARS, m_star, rng)
    print(f"Initial bodies: {sim.N} (including star)")

    # Track water content by initial position
    origins = []
    initial_masses = []
    for p in sim.particles[1:]:
        origins.append(p.a >= r_snow)  # True if formed beyond snow line
        initial_masses.append(p.m / 3e-6)  # Earth masses

    # Create animation
    writer = FFMpegWriter(fps=ANIMATION_FPS, metadata=dict(artist='REBOUND'), bitrate=2000)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    with writer.saving(fig, "enhanced_planet_formation.mp4", dpi=150):
        t = 0.0
        frame_count = 0
        
        while t < SIM_YEARS:
            sim.integrate(t)
            
            # Clear and setup plot
            ax.clear()
            ax.set_facecolor("black")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_xlabel("x [AU]", color='white')
            ax.set_ylabel("y [AU]", color='white')
            ax.tick_params(colors='white')
            
            # Title with current time and body count
            ax.set_title(f"Planet Formation  |  t = {t/1000:.1f} kyr  |  Bodies: {sim.N-1}", 
                        color='white', fontsize=14, pad=20)

            # Plot bodies
            xs, ys, sizes, colors = [], [], [], []
            
            for i, p in enumerate(sim.particles[1:]):  # Skip star
                xs.append(p.x)
                ys.append(p.y)
                
                # Mass in Earth masses
                m_earth = p.m / 3e-6
                
                # Size proportional to mass but visible
                size = np.clip(np.sqrt(m_earth) * 8, 3, 50)
                sizes.append(size)
                
                # Color based on water content and mass
                if origins[i] if i < len(origins) else False:  # Water-rich
                    if m_earth > 0.5:
                        colors.append('lightblue')  # Large water world
                    else:
                        colors.append('cyan')       # Small icy body
                else:  # Rocky
                    if m_earth > 0.5:
                        colors.append('orange')     # Rocky planet
                    else:
                        colors.append('red')        # Small rocky body

            # Plot particles with trails for larger bodies
            scatter = ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.8, 
                               edgecolors='white', linewidths=0.5)

            # Add habitable zone
            hz_inner = plt.Circle((0,0), hz_in, color="lime", fill=False, 
                                alpha=0.6, linewidth=2, linestyle='--')
            hz_outer = plt.Circle((0,0), hz_out, color="lime", fill=False, 
                                alpha=0.6, linewidth=2, linestyle='--')
            ax.add_artist(hz_inner)
            ax.add_artist(hz_outer)
            
            # Add snow line
            snow_circle = plt.Circle((0,0), r_snow, color="lightblue", fill=False,
                                   alpha=0.4, linewidth=1, linestyle=':')
            ax.add_artist(snow_circle)
            
            # Add central star
            ax.scatter([0], [0], s=100, c='yellow', marker='*', 
                      edgecolor='orange', linewidth=1, zorder=10)

            # Legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', 
                          markersize=8, label='Water-rich', alpha=0.8),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                          markersize=8, label='Rocky', alpha=0.8),
                plt.Line2D([0], [0], color='lime', linestyle='--', 
                          label='Habitable Zone', alpha=0.8),
                plt.Line2D([0], [0], color='lightblue', linestyle=':', 
                          label='Snow Line', alpha=0.6)
            ]
            ax.legend(handles=legend_elements, loc='upper right', 
                     facecolor='black', edgecolor='white', labelcolor='white')
            
            # Grid
            ax.grid(True, alpha=0.3, color='gray')
            ax.set_aspect('equal')
            
            writer.grab_frame()
            
            if frame_count % 20 == 0:  # Progress update
                print(f"  Frame {frame_count}: t={t/1000:.1f} kyr, bodies={sim.N-1}")
            
            t += FRAME_INTERVAL_YEARS
            frame_count += 1

        plt.close(fig)

    print(f"\n✅ Animation complete: enhanced_planet_formation.mp4")
    print(f"Final body count: {sim.N-1} (started with {N_EMBRYOS + N_PLANETESIMALS})")
    
    # Final analysis
    final_planets = []
    for i, p in enumerate(sim.particles[1:]):
        m_earth = p.m / 3e-6
        in_hz = hz_in <= p.a <= hz_out
        is_water_rich = origins[i] if i < len(origins) else False
        final_planets.append({
            'mass_earth': m_earth,
            'distance_au': p.a,
            'in_habitable_zone': in_hz,
            'water_rich': is_water_rich,
            'potentially_habitable': in_hz and 0.5 <= m_earth <= 5.0 and is_water_rich
        })
    
    df = pd.DataFrame(final_planets)
    print(f"\nFinal System Analysis:")
    print(f"- Total planets: {len(df)}")
    print(f"- In habitable zone: {df['in_habitable_zone'].sum()}")
    print(f"- Water-rich: {df['water_rich'].sum()}")
    print(f"- Potentially habitable: {df['potentially_habitable'].sum()}")
    
    if len(df) > 0:
        print(f"- Mass range: {df['mass_earth'].min():.3f} - {df['mass_earth'].max():.3f} Earth masses")
        print(f"- Distance range: {df['distance_au'].min():.2f} - {df['distance_au'].max():.2f} AU")

if __name__ == "__main__":
    main()