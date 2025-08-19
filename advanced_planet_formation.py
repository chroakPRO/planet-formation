#!/usr/bin/env python3
"""
Advanced planet formation simulation with dramatic collisions, growth, and migration
Designed for 1-4 hour runtime with spectacular visual results
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

# Advanced simulation config for dramatic planet formation
N_SYSTEMS = 1                   # One detailed system
N_EMBRYOS = 25                  # More starting embryos
N_PLANETESIMALS = 150           # Many collision targets
A_MIN = 0.3                     # Very inner edge (hot zone)
A_MAX = 6.0                     # Extended outer disk
M_STAR = 1.0                    # Solar mass
DT_YEARS = 0.005                # Very small timestep for accuracy
SIM_YEARS = 10e6                # 10 MILLION years - real planet formation time!
FRAME_INTERVAL_YEARS = 25000    # 25,000 years per frame
ANIMATION_FPS = 15              # Smooth video
MAX_FRAMES = 600                # Cap for reasonable file size

# Physical parameters for realistic dynamics
GAS_DRAG_TIMESCALE = 1e5        # Gas disk causes orbital decay
COLLISION_EFFICIENCY = 0.8      # Not all encounters result in mergers
MIN_COLLISION_VELOCITY = 1e-3   # km/s - minimum for fragmentation

def star_luminosity(mstar):
    return mstar**3.5

def habitable_zone(lum):
    return 0.95*sqrt(lum), 1.67*sqrt(lum)

def snow_line(lum):
    return 2.7*sqrt(lum)

class AdvancedCollisionHandler:
    """Custom collision handler for realistic planet formation"""
    
    def __init__(self, sim):
        self.sim = sim
        self.collision_history = []
        self.water_fractions = {}  # Track water content through mergers
        
    def collision_resolve(self, collision):
        """Handle collisions with mass growth and water mixing"""
        p1 = collision.p1
        p2 = collision.p2
        
        # Calculate collision energy
        rel_vel = ((p1.vx - p2.vx)**2 + (p1.vy - p2.vy)**2 + (p1.vz - p2.vz)**2)**0.5
        
        # Record collision
        self.collision_history.append({
            'time': self.sim.t,
            'mass1': p1.m,
            'mass2': p2.m,
            'velocity': rel_vel,
            'combined_mass': p1.m + p2.m
        })
        
        # Merge masses and conserve momentum
        total_mass = p1.m + p2.m
        
        # New position (mass-weighted average)
        new_x = (p1.m * p1.x + p2.m * p2.x) / total_mass
        new_y = (p1.m * p1.y + p2.m * p2.y) / total_mass
        new_z = (p1.m * p1.z + p2.m * p2.z) / total_mass
        
        # New velocity (momentum conservation)
        new_vx = (p1.m * p1.vx + p2.m * p2.vx) / total_mass
        new_vy = (p1.m * p1.vy + p2.m * p2.vy) / total_mass
        new_vz = (p1.m * p1.vz + p2.m * p2.vz) / total_mass
        
        # Update the first particle
        p1.m = total_mass
        p1.x, p1.y, p1.z = new_x, new_y, new_z
        p1.vx, p1.vy, p1.vz = new_vx, new_vy, new_vz
        
        # Mix water fractions
        id1, id2 = id(p1), id(p2)
        water1 = self.water_fractions.get(id1, 0.0)
        water2 = self.water_fractions.get(id2, 0.0)
        self.water_fractions[id1] = (p1.m * water1 + p2.m * water2) / total_mass
        if id2 in self.water_fractions:
            del self.water_fractions[id2]
        
        # Remove the second particle
        return 1  # Remove particle 2

def apply_gas_drag(sim, timescale_years):
    """Apply gas disk drag causing orbital migration"""
    for i, particle in enumerate(sim.particles[1:], 1):  # Skip star
        if sim.t < 3e6:  # Gas disk dissipates after 3 Myr
            # Drag causes orbital decay
            decay_rate = 1.0 / (timescale_years * 2 * np.pi)  # Convert to simulation units
            
            # Apply drag to velocity (causes spiral inward)
            drag_factor = 1.0 - decay_rate * sim.dt
            particle.vx *= drag_factor
            particle.vy *= drag_factor
            particle.vz *= drag_factor

def build_advanced_sim(n_planetesimals, n_embryos, a_min, a_max, dt, m_star, rng):
    """Build simulation with realistic mass distribution and dynamics"""
    sim = rebound.Simulation()
    sim.units = ("AU","yr","Msun")
    sim.add(m=m_star)  # Central star
    
    # Realistic mass distribution (power law)
    # Larger embryos in inner system, smaller in outer
    
    # Embryos - proto-planets with realistic mass distribution
    for i in range(n_embryos):
        # Inner system gets more massive embryos
        a = rng.uniform(a_min, a_max)
        mass_factor = (a_max / a)**0.5  # More massive closer in
        base_mass = rng.uniform(0.05, 0.8) * mass_factor
        mass = base_mass * 3e-6  # Convert to solar masses
        
        # Realistic orbital elements with some chaos
        e = rng.uniform(0.01, 0.15)  # Moderate eccentricities
        inc = rng.uniform(0, 3.0) * np.pi/180  # Small inclinations
        
        # Random orbital phases
        Omega = rng.uniform(0, 2*np.pi)
        omega = rng.uniform(0, 2*np.pi)
        f = rng.uniform(0, 2*np.pi)
        
        sim.add(m=mass, a=a, e=e, inc=inc, Omega=Omega, omega=omega, f=f)
    
    # Planetesimals - many small bodies
    for i in range(n_planetesimals):
        # Spread throughout disk with realistic size distribution
        a = rng.uniform(a_min, a_max)
        
        # Smaller masses, power-law distribution
        mass_exp = rng.uniform(-2.5, -1.5)  # Size distribution
        base_mass = 10**mass_exp * 0.1  # Earth masses
        mass = base_mass * 3e-6  # Convert to solar masses
        
        # Small random eccentricities and inclinations
        e = rng.uniform(0.005, 0.08)
        inc = rng.uniform(0, 2.0) * np.pi/180
        
        Omega = rng.uniform(0, 2*np.pi)
        omega = rng.uniform(0, 2*np.pi)
        f = rng.uniform(0, 2*np.pi)
        
        sim.add(m=mass, a=a, e=e, inc=inc, Omega=Omega, omega=omega, f=f)
    
    # Setup integrator and collisions
    sim.move_to_com()
    sim.integrator = "whfast"
    sim.dt = dt
    sim.collision = "direct"
    sim.collision_resolve = "merge"
    
    # Add custom physics
    collision_handler = AdvancedCollisionHandler(sim)
    
    return sim, collision_handler

def main():
    print(f"🚀 Advanced Planet Formation Simulation")
    print(f"========================================")
    print(f"Simulation time: {SIM_YEARS/1e6:.1f} million years")
    print(f"Frame interval: {FRAME_INTERVAL_YEARS/1000:.0f} thousand years")
    print(f"Expected frames: {min(int(SIM_YEARS/FRAME_INTERVAL_YEARS), MAX_FRAMES)}")
    print(f"Video length: ~{min(int(SIM_YEARS/FRAME_INTERVAL_YEARS), MAX_FRAMES)/ANIMATION_FPS:.1f} seconds")
    print(f"Estimated runtime: 1-4 hours (depending on collisions)")
    
    rng = np.random.default_rng(42)
    m_star = M_STAR
    lum = star_luminosity(m_star)
    hz_in, hz_out = habitable_zone(lum)
    r_snow = snow_line(lum)

    print(f"\n⭐ Star System Parameters:")
    print(f"   Star mass: {m_star:.1f} solar masses")
    print(f"   Habitable zone: {hz_in:.2f} - {hz_out:.2f} AU")
    print(f"   Snow line: {r_snow:.2f} AU")
    print(f"   Disk extent: {A_MIN:.1f} - {A_MAX:.1f} AU")

    # Build advanced simulation
    sim, collision_handler = build_advanced_sim(N_PLANETESIMALS, N_EMBRYOS, A_MIN, A_MAX, DT_YEARS, m_star, rng)
    print(f"\n🌌 Initial Conditions:")
    print(f"   Total bodies: {sim.N-1} (excluding star)")
    print(f"   Embryos: {N_EMBRYOS}")
    print(f"   Planetesimals: {N_PLANETESIMALS}")

    # Initialize water content tracking
    for i, p in enumerate(sim.particles[1:]):
        # Bodies beyond snow line start water-rich
        water_fraction = 0.5 if p.a >= r_snow else 0.05
        collision_handler.water_fractions[id(p)] = water_fraction

    # Create high-quality animation
    writer = FFMpegWriter(fps=ANIMATION_FPS, metadata=dict(artist='Advanced REBOUND'), bitrate=3000)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Track evolution statistics
    body_count_history = []
    collision_count_history = []
    mass_history = []
    
    start_time = time.time()
    
    with writer.saving(fig, "advanced_planet_formation.mp4", dpi=120):
        t = 0.0
        frame_count = 0
        next_report = SIM_YEARS / 20  # Progress reports
        
        while t < SIM_YEARS and frame_count < MAX_FRAMES:
            # Integrate physics
            sim.integrate(t)
            
            # Apply gas drag (orbital migration)
            apply_gas_drag(sim, GAS_DRAG_TIMESCALE)
            
            # Clear plots
            ax1.clear()
            ax2.clear()
            
            # === MAIN ORBITAL VIEW ===
            ax1.set_facecolor("black")
            ax1.set_xlim(-7, 7)
            ax1.set_ylim(-7, 7)
            ax1.set_xlabel("x [AU]", color='white', fontsize=12)
            ax1.set_ylabel("y [AU]", color='white', fontsize=12)
            ax1.tick_params(colors='white')
            
            # Dynamic title with statistics
            collisions_total = len(collision_handler.collision_history)
            ax1.set_title(f"Planet Formation  |  {t/1e6:.2f} Myr  |  Bodies: {sim.N-1}  |  Collisions: {collisions_total}", 
                         color='white', fontsize=14, pad=15)

            # Plot bodies with enhanced visualization
            xs, ys, sizes, colors, alphas = [], [], [], [], []
            
            largest_mass = 0
            for i, p in enumerate(sim.particles[1:]):  # Skip star
                xs.append(p.x)
                ys.append(p.y)
                
                # Mass in Earth masses
                m_earth = p.m / 3e-6
                largest_mass = max(largest_mass, m_earth)
                
                # Dynamic sizing based on mass
                size = np.clip(np.log10(m_earth + 0.001) * 15 + 25, 5, 80)
                sizes.append(size)
                
                # Advanced color coding based on composition and mass
                water_frac = collision_handler.water_fractions.get(id(p), 0.05)
                
                if m_earth > 1.0:  # Large planets
                    if water_frac > 0.2:
                        colors.append('lightblue')  # Water world
                        alphas.append(0.9)
                    else:
                        colors.append('orange')     # Rocky planet
                        alphas.append(0.9)
                elif m_earth > 0.1:  # Medium bodies
                    if water_frac > 0.2:
                        colors.append('cyan')       # Icy body
                        alphas.append(0.8)
                    else:
                        colors.append('red')        # Rocky body
                        alphas.append(0.8)
                else:  # Small bodies
                    if water_frac > 0.2:
                        colors.append('lightcyan')  # Ice fragments
                        alphas.append(0.6)
                    else:
                        colors.append('brown')      # Rock fragments
                        alphas.append(0.6)

            # Plot particles with glowing effect
            for x, y, s, c, a in zip(xs, ys, sizes, colors, alphas):
                ax1.scatter([x], [y], s=s*1.5, c='white', alpha=0.3, zorder=1)  # Glow
                ax1.scatter([x], [y], s=s, c=c, alpha=a, edgecolors='white', 
                           linewidths=0.5, zorder=2)

            # Add zone indicators
            hz_inner = plt.Circle((0,0), hz_in, color="lime", fill=False, 
                                alpha=0.7, linewidth=2, linestyle='--', label='Habitable Zone')
            hz_outer = plt.Circle((0,0), hz_out, color="lime", fill=False, 
                                alpha=0.7, linewidth=2, linestyle='--')
            ax1.add_artist(hz_inner)
            ax1.add_artist(hz_outer)
            
            snow_circle = plt.Circle((0,0), r_snow, color="lightblue", fill=False,
                                   alpha=0.5, linewidth=2, linestyle=':', label='Snow Line')
            ax1.add_artist(snow_circle)
            
            # Central star with dynamic brightness
            star_brightness = 150 + 50 * np.sin(t / 1e5)  # Slight variability
            ax1.scatter([0], [0], s=star_brightness, c='yellow', marker='*', 
                       edgecolor='orange', linewidth=2, zorder=10, alpha=0.9)

            ax1.grid(True, alpha=0.2, color='gray')
            ax1.set_aspect('equal')
            
            # === STATISTICS PANEL ===
            ax2.set_facecolor("black")
            ax2.set_xlim(0, 10)
            ax2.set_ylim(0, 10)
            ax2.axis('off')
            
            # Current statistics
            total_mass = sum(p.m / 3e-6 for p in sim.particles[1:])
            bodies_in_hz = sum(1 for p in sim.particles[1:] if hz_in <= p.a <= hz_out)
            
            stats_text = f"""SYSTEM EVOLUTION STATISTICS
            
Time Elapsed: {t/1e6:.2f} / {SIM_YEARS/1e6:.1f} Myr
Progress: {t/SIM_YEARS*100:.1f}%

CURRENT STATE:
• Bodies Remaining: {sim.N-1}
• Bodies in HZ: {bodies_in_hz}
• Total Mass: {total_mass:.2f} M⊕
• Largest Body: {largest_mass:.3f} M⊕
• Total Collisions: {collisions_total}

COMPOSITION LEGEND:
🔵 Large Water World (>1 M⊕, water-rich)
🟠 Large Rocky Planet (>1 M⊕, dry)
🔷 Medium Icy Body (0.1-1 M⊕, water-rich)
🔴 Medium Rocky Body (0.1-1 M⊕, dry)
⚪ Small Fragments (<0.1 M⊕)

ZONES:
▬ ▬ Habitable Zone (liquid water)
· · · Snow Line (water freezes beyond)
"""
            
            ax2.text(0.5, 9.5, stats_text, color='white', fontsize=11, 
                    verticalalignment='top', horizontalalignment='left',
                    fontfamily='monospace')
            
            # Track history
            body_count_history.append(sim.N-1)
            collision_count_history.append(collisions_total)
            mass_history.append(total_mass)
            
            # Save frame
            writer.grab_frame()
            
            # Progress reporting
            if t >= next_report:
                elapsed = time.time() - start_time
                progress = t / SIM_YEARS
                eta = elapsed / progress - elapsed if progress > 0 else 0
                print(f"  Progress: {progress*100:.1f}% | t={t/1e6:.2f} Myr | Bodies={sim.N-1} | "
                      f"Collisions={collisions_total} | ETA: {eta/3600:.1f}h")
                next_report += SIM_YEARS / 20
            
            t += FRAME_INTERVAL_YEARS
            frame_count += 1

        plt.close(fig)

    runtime = time.time() - start_time
    print(f"\n🎬 Animation Complete!")
    print(f"   File: advanced_planet_formation.mp4")
    print(f"   Runtime: {runtime/3600:.2f} hours")
    print(f"   Frames: {frame_count}")
    print(f"   Final bodies: {sim.N-1} (started with {N_EMBRYOS + N_PLANETESIMALS})")
    print(f"   Total collisions: {len(collision_handler.collision_history)}")
    
    # Final detailed analysis
    final_planets = []
    habitable_candidates = []
    
    for i, p in enumerate(sim.particles[1:]):
        m_earth = p.m / 3e-6
        in_hz = hz_in <= p.a <= hz_out
        water_frac = collision_handler.water_fractions.get(id(p), 0.05)
        is_rocky_planet = 0.5 <= m_earth <= 10.0
        is_water_rich = water_frac > 0.2
        
        potentially_habitable = in_hz and is_rocky_planet and is_water_rich
        
        planet_data = {
            'mass_earth': m_earth,
            'distance_au': p.a,
            'eccentricity': p.e,
            'water_fraction': water_frac,
            'in_habitable_zone': in_hz,
            'is_rocky_size': is_rocky_planet,
            'water_rich': is_water_rich,
            'potentially_habitable': potentially_habitable
        }
        
        final_planets.append(planet_data)
        if potentially_habitable:
            habitable_candidates.append(planet_data)
    
    df = pd.DataFrame(final_planets)
    df.to_csv("advanced_planet_formation_results.csv", index=False)
    
    print(f"\n🌍 Final System Analysis:")
    print(f"   Total planets: {len(df)}")
    print(f"   Rocky planets (0.5-10 M⊕): {df['is_rocky_size'].sum()}")
    print(f"   In habitable zone: {df['in_habitable_zone'].sum()}")
    print(f"   Water-rich planets: {df['water_rich'].sum()}")
    print(f"   Potentially habitable: {df['potentially_habitable'].sum()}")
    
    if len(df) > 0:
        print(f"   Mass range: {df['mass_earth'].min():.3f} - {df['mass_earth'].max():.3f} M⊕")
        print(f"   Distance range: {df['distance_au'].min():.2f} - {df['distance_au'].max():.2f} AU")
        
        if len(habitable_candidates) > 0:
            print(f"\n🎯 Habitable Planet Candidates:")
            for i, planet in enumerate(habitable_candidates):
                print(f"     Planet {i+1}: {planet['mass_earth']:.2f} M⊕ at {planet['distance_au']:.2f} AU "
                      f"(water: {planet['water_fraction']*100:.1f}%)")

if __name__ == "__main__":
    main()