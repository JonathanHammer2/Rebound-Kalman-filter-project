# This script is written to derive the H matrix at a moment
# When simulate to the past, H matrix would be updated with time

# This script is to simulate the Earth's eccentricity.
# General relativity and Solar quadrupole are added.
# The Earth-Moon system is treated as a quadrupole.

# Configuration   coder   date      change
#    --           YW       4/23     --
#     A           JH       9/19/23  Reversed earth and mecury, venus to put inner planets first in models

import matplotlib.pyplot as plt
import numpy as np
import rebound
import reboundx
from reboundx import constants
import time
import pandas as pd


def main():
    # Part Zero: These inputs can be adjusted as needed. =============
    tKyr=-100 # in kyr
    step=-2 * np.pi * 0.035 # intergation step
    delta_t=  -np.pi * 0.035#-2 * np.pi * 1e-6 # the threshold is 1e-6 to get stable e_dot


    figname = "Fig_" + str(tKyr) + ".jpg" # Name of the figure
    saveFile='Sim_Rebound'+str(tKyr)+'.csv' # Name of the solution file
    # title='Eccentricity and e_dot' # Title of the figure

    # Part I: set simulation =========================================
    # Some astronomical parameters ====
    AU = 149597870.7  # AU in unit "km"
    M = 1 / 332946.048773  # Earth's mass
    m = 1 / 27068702.952351  # Moon's mass
    J2 = 0.5 * 0.8525 * M * m / (M + m) ** 2 # J2=0.005116256454851977
        # J2=0.5*fMm/(M+m)^2, where f=0.8525

    date = "2000-12-31 00:00"
    solar_system_objects = ["Sun", "Mercury", "Venus", "Earth", "Mars",
                            "Jupiter", "Saturn", "Uranus", "Neptune"]

    sim_2p=[] # The "2p" means interactions between two planets.

    for i in range(8):
        sim_2p.append(rebound.Simulation())

    ps_2p = []  # A list to save particles of sim_2p
    rebx_2p = [] # A list to save reboundx of sim_2p

    # Add objects
    sim_2p[0].add(["Sun", "Mercury", "Earth" ], date=date)  # reversed venus, mercury and earth, JH 9-19-23
    sim_2p[1].add(["Sun", "Venus", "Earth" ], date=date)
    sim_2p[2].add(solar_system_objects,date=date) # a place holder
    sim_2p[3].add(["Sun", "Earth", "Mars", ], date=date)
    sim_2p[4].add(["Sun", "Earth", "Jupiter", ], date=date)
    sim_2p[5].add(["Sun", "Earth", "Saturn", ], date=date)
    sim_2p[6].add(["Sun", "Earth", "Uranus", ], date=date)
    sim_2p[7].add(["Sun", "Earth", "Neptune", ], date=date)


    for i in range(8):
        if i < 2:
            sim_2p[i].particles[2].hash = 'Earth'
        elif i == 2:
            sim_2p[i].particles[3].hash = 'Earth'  # Add a hash tag
        else:
            sim_2p[i].particles[1].hash = 'Earth'  # Add a hash tag

        sim_2p[i].move_to_com()
        ps_2p.append(sim_2p[i].particles)
        sim_2p[i].integrator = "whfast"
        sim_2p[i].ri_whfast.corrector = 17
        sim_2p[i].ri_whfast.safe_mode = 0
        sim_2p[i].dt = step #  dt, 0.035 yr (12.8 d)

        # ======================================================
        rebx_2p.append(reboundx.Extras(sim_2p[i]))

        # Add General Relativity --------
        gr = rebx_2p[i].load_force("gr") # "gr_full" has the same result
        gr.params["c"] = constants.C # Set light speed
        rebx_2p[i].add_force(gr) # Add GR as a force

        # Add Gravitational Harmonics --------
        gh = rebx_2p[i].load_force("gravitational_harmonics")
        rebx_2p[i].add_force(gh)

        # Add the solar quadrupole --------
        ps_2p[i][0].params["J2"] = 2.2e-7        # Sun
        ps_2p[i][0].params["R_eq"] = 695700/AU   # Sun

        # Add the Earth-Moon quadrupole --------
        ps_2p[i]['Earth'].params["J2"] = J2             # Earth-Moon system
        ps_2p[i]['Earth'].params["R_eq"] = 3.844e5/AU   # Earth-Moon system
        # ======================================================

    # The sim is the main integration, with Sun and 8 planets
    sim=sim_2p[2]
    ps=sim_2p[2].particles
    # rebx=rebx_2p[2]

    #xyz


    # Part II: start to simulation =====================================
    # 2.1 Set output points ========
    Nout = int(-tKyr / 0.1 + 1)  # Number of output points for plot, e.g. 1000
    times = np.linspace(0, tKyr * 1000 * (2.) * np.pi, Nout)

    ecc0=ps['Earth'].calculate_orbit(primary=ps[0]).e
    print('Time=0, ecc=',ecc0)
    ecc_lst = []
    e_dot_lst=[]
    e_dot_lst.append(0) # jh

    # 2.2 Integrate and record ========
    start_time=time.time() # Start timeing
    tlast = start_time
    ecclast = ecc0

    for ti, t in enumerate(times):
        sim.integrate(t)  # Integrate
        ecc = ps['Earth'].calculate_orbit(primary=ps[0]).e # Calculate eccentricity
        ecc_lst.append(ecc)

        H_mat = []
        for j in range(0, 8):
            if j!=2:
                # Set time for planet interaction integration
                sim_2p[j].t = sim.t

                # Set objects for planet interaction integration
                ps_2p[j][0]  = ps[0] # Sun
                ps_2p[j][1]  = ps['Earth'] # Earth
                ps_2p[j][2]  = ps[j + 1]  # the other planet

                ecc11 = ps_2p[j]['Earth'].calculate_orbit(primary=ps_2p[j][0]).e
                sim_2p[j].integrate(t+delta_t) # integrate a tiny step
                ecc22 = ps_2p[j]['Earth'].calculate_orbit(primary=ps_2p[j][0]).e
                dee = ecc22 - ecc11 # delta ecc
                e_dott = dee/delta_t # ecc dot

                # In the H matrix, I use partial ecc over partial state (x,vx,y,vy,z,vz),
                # because the partial e_dot is too unstable
                # The denominator has already divided by delta_t,
                # because of the integration from t to t+delta_t.
                H_mat.append(e_dott / (ps_2p[j][2].x  - ps[j+1].x ))
                H_mat.append(e_dott / (ps_2p[j][2].vx - ps[j+1].vx))
                H_mat.append(e_dott / (ps_2p[j][2].y  - ps[j+1].y ))
                H_mat.append(e_dott / (ps_2p[j][2].vy - ps[j+1].vy))
                H_mat.append(e_dott / (ps_2p[j][2].z  - ps[j+1].z ))
                H_mat.append(e_dott / (ps_2p[j][2].vz - ps[j+1].vz))

   #     print(H_mat)

        # get e_dot, for output
        #sim.integrate(t+delta_t)
        #ecc2 = ps['Earth'].calculate_orbit(primary=ps[0]).e
        #e_dot = (ecc2 - ecc)/delta_t


        #if ti == 0:
        #    e_dot_lst.append(0)
        if ti == 1:
            e_dot = (ecc - ecclast) / (t - tlast)
            e_dot_lst.append(e_dot)
            tlast = t
            ecclast = ecc
        elif ti > 1:
            #e_dot = e_dot_lst[ti-1] + 0.1*(ecc_lst[ti] - ecc_lst[ti-1]) / (times[ti] - times[ti-1])
            e_dot = e_dot + 0.1*(ecc - ecclast) / (t - tlast)
            e_dot_lst.append(e_dot)
            tlast = t
            ecclast = ecc

        # if (ti + 1) % int(0.1 * Nout) == 0:  # Monitor the process
        #     print('Progress(%):', (ti + 1) // int(0.1 * Nout) * 10)

    print('Time consumed:', time.time() - start_time)  # End timing

    # 2.3 Output time, eccentricity, and e_dot ========
    StoreStuff = np.array([times / (2 * np.pi * 1e3), ecc_lst,e_dot_lst])
    np.savetxt(saveFile, StoreStuff.T, delimiter=',')  # Save the solution

    # Part III: plot ===================================================
    # 3.1 Plot the simulated eccentricity ========
    plt.figure(1, figsize=(16, 6))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.9)
    plt.subplot(2,1,1)
    plt.plot(times / (2 * np.pi * 1e3), ecc_lst, label='Rebound')

    # 3.2 Plot Laskar and/or Zeebe's models ========
    ZB20a = pd.read_csv('./ZB20a.csv').values  # two colums, time and ecc
    plt.plot(ZB20a[:, 0], ZB20a[:, 1], label='ZB20a')
    plt.xlim([tKyr, 0])  # Change the xlim if needed.
    # plt.title(title)
    plt.ylabel('Eccentricity')
    plt.legend(loc='best')

    # 3.3 Plot the simulated e_dot (too unstable) ========
    plt.subplot(2, 1, 2)
    plt.plot(times / (2 * np.pi * 1e3), np.array(e_dot_lst)*2*np.pi)
    plt.xlim([tKyr, 0])  # Change the xlim if needed.
    plt.ylabel('E_dot (yr$^{-1}$)')
    plt.xlabel('Time(kyr)')

    plt.savefig(figname, dpi=300)
    plt.show()

    a = 5


if __name__ == '__main__':
    main()