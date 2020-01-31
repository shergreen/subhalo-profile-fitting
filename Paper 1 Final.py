#!/usr/bin/env python
# coding: utf-8

# # Density Profile Fitting $\rho(r| f_\mathrm{b}, c_\mathrm{s})$
# 
# Note: Additional analysis cells can be found in the full backup of this notebook, titled "Density Profile Fitting"; this version aims to provide the minimal reproduction of all figures from Green & van den Bosch (2019).

# ## Data load-in and preprocessing

# In[1]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('xmode', 'Verbose')
#Context
import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
from pathlib import Path
from useful_functions.plotter import plot, loglogplot
from scipy.optimize import ridder, fsolve, leastsq, root, minimize, curve_fit, basinhopping
import scipy.integrate as integrate
from numba import jit, njit, prange
from matplotlib.ticker import MultipleLocator
import emcee
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.integrate import quad
from multiprocessing import Pool
#from matplotlib2tikz import save as save_tikz
from matplotlib import animation, rc
from IPython.display import HTML
from os.path import expanduser


# In[2]:


home_dir = Path(expanduser('~'))


# In[3]:


fig_dir = home_dir / 'Drive/Research/substructure/tidal_evolution/'


# In[4]:


#write a function that will generate a plot of the bound fraction and density profile evolution given the param sets
def plot_fb_rho(ch_num, cs_num, xc_num, eta_num, normalize=False,by_iso=False):
    direct = dash_root / ch_dirs[ch_num] / cs_dirs[cs_num] / xc_dirs[xc_num] / eta_dirs[eta_num]
    pf_fn = direct / 'radprof_rho.txt'
    sh_evo_fn = direct / 'subhalo_evo.txt'
    if(Path(pf_fn).is_file()):
        sim_dat = load_sim(direct,cs_num,False,normalize,by_iso) #keep all timesteps
        fig, ax = plot()
        plt.plot(times[0:sim_dat.shape[0]],sim_dat[:,0])
        plt.xlabel('$t$ (Gyr)')
        plt.ylabel('$f_b$')
        fig, ax = loglogplot()
        plt.xlabel(r'$r/r_\textrm{s,0}$')
        plt.ylabel(r'$\rho(r) / \rho(r,t=0)$')
        plt.title('Subhalo with $c_h$=%.2f, $c_s$=%.2f, $x_c$=%.2f, $\eta$=%.2f' %(ch_vals[ch_num], cs_vals[cs_num], xc_vals[xc_num], eta_vals[eta_num]))
        plt.xlim(np.min(radii*cs_vals[cs_num]),cs_vals[cs_num])
        plt.ylim(10**-1,2.)
        for i in range(0,sim_dat.shape[0]):
            plt.plot(radii*cs_vals[cs_num],sim_dat[i,1:] / sim_dat[0,1:],color=cols[i])
        
    else:
        print("No simulation run here yet!")
    
    return fig, ax
        
def plot_fb_rho_iso(cs_num):
    direct = iso_root / cs_dirs[cs_num]
    pf_fn = direct / 'radprof_rho.txt'
    sh_evo_fn = direct / 'subhalo_evo.txt'
    if(Path(pf_fn).is_file()):
        sim_dat = load_sim(direct,cs_num,False,normalize=False) #keep all timesteps, don't throw out near pericenter
        fig, ax = plot()
        plt.plot(times[0:sim_dat.shape[0]],sim_dat[:,0])
        plt.xlabel('$t$ (Gyr)')
        plt.ylabel('$f_b$')
        fig, ax = loglogplot()
        plt.xlabel(r'$r/r_\textrm{s,0}$')
        plt.ylabel(r'$\rho(r) / \rho(r,t=0)$')
        #plt.xlim(0.8*1.122E-3*cs_vals[cs_num],1.2*cs_vals[cs_num])
        plt.xlim(np.min(radii*cs_vals[cs_num]),cs_vals[cs_num])
        plt.axhline(1., zorder=30)
        plt.ylim(10**-1,2.)
        plt.title('Isolated Subhalo with $c_s=$%.2f \nBright to Dark represents Time Evolution' % cs_vals[cs_num])
        for i in range(0,sim_dat.shape[0]):
            plt.plot(radii*cs_vals[cs_num],sim_dat[i,1:] / sim_dat[0,1:],color=cols[i])
        plt.savefig('isolated_subhalo_cs%.2f.pdf' % cs_vals[cs_num],bbox_inches='tight')
        
    else:
        print("No simulation run here yet!")


# In[5]:


dash_root = home_dir / 'new_DASH/' # fixed ICs re-run by Sheridan at YCRC
iso_root = dash_root / 'iso/'
iso_dec_root = dash_root / 'iso_exp_dec/'
ic_dir = dash_root / 'ic/'
eta_vals = np.linspace(0,1,11)
xc_vals = np.logspace(np.log10(0.5),np.log10(2.0),11)
cs_vals = np.around(np.logspace(np.log10(3.149),np.log10(31.5),11),1)
ch_vals = np.around(np.logspace(np.log10(3.149),np.log10(31.5),11),1)
#3.1,4.0,5.0,6.3,7.9,10.0,12.5,15.8,19.9,25.0,31.5
eta_dirs = []
ch_dirs = []
cs_dirs = []
xc_dirs = []
for i in range(0,11):
    eta_dirs.extend(['eta%d/' %i])
    xc_dirs.extend(['xc%d/' %i])
    ch_dirs.extend(['ch%d/' %i])
    cs_dirs.extend(['cs%d/' %i])


# In[6]:


#make directories
#loop over them and store the relevant values
dr = []
for i in range(0,11):
    for j in range(0,11):
        for k in range(0,11):
            for l in range(0,11):
                dr.extend([dash_root / ch_dirs[i] / cs_dirs[j] / xc_dirs[k] / eta_dirs[l]])


# In[7]:


#relevant constants for the sim
Npart = 1048576.
rho200_0 = 1.0 / (4.0 * np.pi / 3.0) #0.237 #in model units (its 0.2387.. so we get H=0.1) (it's really 1.0 / (4.0 * pi / 3.0))
#this value normalizes things such that rho200_0 * (rho/rho_200) * V_shell = 1, and m_p = 1./Npart
mp = 1. / Npart #particle mass
Ncrit = 1280. #from Ncrit = 80*Nb**0.2 (t=0)
sim_eps = 0.0003 #in units of r_vir,s
G=1


# In[8]:


timesteps = 301
nsims = 11**4
times = np.linspace(0,36,timesteps) * 1. / 1.44 #converted from Gyr to model units
delta_t = times[1] - times[0]
outermost_radius = 10. #in units of Rvir
innermost_radius = 10**-3
#the following radii are those used for the density profile (and velocity dispersion)
radii = np.loadtxt(dash_root / ch_dirs[0] / cs_dirs[10] / xc_dirs[5] / eta_dirs[5] / 'radprof_rho.txt')[0,1:]
mass_prof_radii = 10**((np.log10(radii[1:]) + np.log10(radii[:-1])) / 2.)
mass_prof_radii = np.append(mass_prof_radii, outermost_radius)
shell_vols = 4.0*np.pi / 3.0 * (mass_prof_radii[1:]**3 - mass_prof_radii[:-1]**3)
shell_vols = np.insert(shell_vols, 0, 4.0*np.pi / 3.0 * (mass_prof_radii[0]**3 - innermost_radius**3))
n_profile_pts = len(radii)
sns.palplot(sns.cubehelix_palette(timesteps))
cols = sns.cubehelix_palette(timesteps)


# In[9]:


#generate final isolated profiles... talk with Frank about what time point we should use for the isolated profiles
#do we want to use the 36 Gyr one? Or do we want to use an earlier timestep
#do we only want to use it in the outer radii?
isolated_final_profs = np.zeros((len(cs_vals), timesteps, len(radii)))

for i in range(0,len(cs_vals)):
    iso_prof_dat = np.loadtxt(iso_root / cs_dirs[i] / 'radprof_rho.txt')[1:,1:]
    isolated_final_profs[i,:,:] = iso_prof_dat #may change which timestep we choose here...
    #do these all satisfy the numerical convergence criteria?


# In[10]:


#to-do: code up the criteria to verify whether or not a given simulation passes the requirements
#will need to use this to filter out the bad ones that we don't want to use

def NFWf(c):
    return np.log(1. + c) - c/(1. + c)

def NFWrho(r, cs, mvir=1.):  # in units where Rvir=1, so r_s = 1/c_s
    rhos = mvir / (4*np.pi * cs**-3 * NFWf(cs))  # this factor drops out...
    return rhos / ((r*cs)*(1. + r*cs)**2)

nfw_profs = np.zeros((len(cs_vals), len(radii)))

for i in range(0, len(cs_vals)):
    for j in range(0, len(radii)):
        nfw_profs[i,j] = NFWrho(radii[j], cs_vals[i]) / rho200_0

chi_crit = 1.79

def convergence_criteria(sh_evo, conc):
    #takes in sh_evo matrix and returns an integer which corresponds to which snapshot the simulation breaks down
    #col 7 of sh_evo is f_b, so just need to check f_b * Npart > Ncrit
    mask1 = sh_evo[:,7] * Npart  > Ncrit
    mask2 = sh_evo[:,7] * NFWf(conc) / (chi_crit * sh_evo[:,8] * sim_eps * conc**2) > 1 #disrupts when ratio below 1
    #print(sh_evo[:,7] / (1.2* sh_evo[:,8] * sim_eps * conc**2))
    #for mask2, see eqns 7,8 from DASH paper
    combined_mask = mask1*mask2
    if(np.all(combined_mask == True)):
        return timesteps #the whole thing is converged, might chance this to 301
    else:
        return(np.min(np.where(combined_mask == False))) #returns first index where not satisfied


#todo: verify that the mask2 calculation is correct, try with more strenuous edge cases to verify
#pretty sure that it is correct, but can go over with Frank my calculation to make sure


# In[11]:


# let's compute what the average densities should be within each NFW cell...
full_mass_prof_radii = np.insert(mass_prof_radii, 0, innermost_radius)
print(full_mass_prof_radii.shape)

# now, we can compute the average density within each NFW cell
avg_nfw_profs = np.zeros((len(cs_vals), len(radii)))

for i in range(0, len(cs_vals)):
    for j in range(0, len(radii)):
        avg_nfw_profs[i,j] = (1.0 / shell_vols[j]) * quad(lambda x: 4*np.pi * x**2 * NFWrho(x, cs=cs_vals[i]), full_mass_prof_radii[j], full_mass_prof_radii[j+1])[0] / rho200_0


# In[12]:


isolated_final_tfs = np.zeros((len(cs_vals), timesteps, len(radii)))

for i in range(0,len(cs_vals)):
    isolated_final_tfs[i,:,:] = isolated_final_profs[i,:,:] / avg_nfw_profs[i,:] #may change which timestep we choose here...
    #do these all satisfy the numerical convergence criteria?


# In[13]:


#we're trying to remove the inner 10% of the orbit close to pericenter to avoid potential periods where the
#subhalo is going through extreme revirialization that may add a lot of variance to the density profile

#the periods will decrease, so if we want to conservative, we should use the first N-1 periods on the first N-1
#pericenters, and then use the N-1th period to determine how much to remove from the Nth pericenter

def sim_rads(xyz):
    #takes the xyz coords and returns radii
    return np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)

'''
def mask_pericenters(sh_evo_dat):
    #TODO: remove 10% of the orbit length based on EACH ORBIT instead of an average, because self-friction
    #reduces the period length rapidly
    #takes the subhalo_evo and returns the snapshot numbers to remove that are near pericenter
    orbital_rads = sim_rads(sh_evo_dat[:,1:4])
    drdt = orbital_rads[1:] - orbital_rads[:-1]
    pericenters = np.where(np.logical_and(drdt[1:] > 0, drdt[:-1] < 0))[0] + 1 #to line up correctly, add one
    orbit_lengths = np.mean(pericenters[1:] - pericenters[:-1]) #or could do sign_flips[1]-sign_flips[0]
    num_snaps_to_remove = int(np.round(orbit_lengths*0.05)) #so that we remove 10% of orbit total
    #remove this many from both sides of the orbit length, see how it varies
    mask = np.ones(len(sh_evo_dat),dtype=bool)
    for pcs in pericenters:
        mask[pcs-num_snaps_to_remove:pcs+num_snaps_to_remove+1] = 0 #mask this one
    return mask
'''

def mask_pericenters(sh_evo_dat):
    #TODO: remove 10% of the orbit length based on EACH ORBIT instead of an average, because self-friction
    #reduces the period length rapidly
    #takes the subhalo_evo and returns the snapshot numbers to remove that are near pericenter
    orbital_rads = sim_rads(sh_evo_dat[:,1:4])
    drdt = orbital_rads[1:] - orbital_rads[:-1]
    pericenters = np.where(np.logical_and(drdt[1:] > 0, drdt[:-1] < 0))[0] + 1 #to line up correctly, add one
    orbit_lengths = pericenters[1:] - pericenters[:-1] #or could do sign_flips[1]-sign_flips[0]
    num_snaps_to_remove = np.round(orbit_lengths*0.05) #so that we remove 10% of orbit total
    num_snaps_to_remove = np.append(num_snaps_to_remove,num_snaps_to_remove[-1]).astype(int)
    #remove this many from both sides of the orbit length, see how it varies
    mask = np.ones(len(sh_evo_dat),dtype=bool)
    for k,pcs in enumerate(pericenters):
        mask[pcs-num_snaps_to_remove[k]:pcs+num_snaps_to_remove[k]+1] = 0 #mask this one
    return mask

def mask_apocenters(sh_evo_dat):
    #finds only the time points at apocenter and returns a mask of those times
    orbital_rads = sim_rads(sh_evo_dat[:,1:4])
    drdt = orbital_rads[1:] - orbital_rads[:-1]
    apocenters = np.where(np.logical_and(drdt[1:] < 0, drdt[:-1] > 0))[0] + 1 #to line up correctly, add one
    mask = np.zeros(len(sh_evo_dat),dtype=bool)
    mask[0] = 1 #count first point
    for k,pcs in enumerate(apocenters):
        mask[apocenters] = 1 # this one is good
    return mask
    
#finding pericentric passage and removing it
test_data = np.loadtxt(dash_root / ch_dirs[0] / cs_dirs[10] / xc_dirs[5] / eta_dirs[5] / 'subhalo_evo.txt')
#test_data = np.loadtxt(dash_root + ch_dirs[2] + cs_dirs[5] + xc_dirs[2] + eta_dirs[0]+'subhalo_evo.txt')[0:80]
#test_data = load_sim(dash_root + ch_dirs[2] + cs_dirs[5] + xc_dirs[2] + eta_dirs[0], 5)
orbital_rads = sim_rads(test_data[:,1:4])
drdt = orbital_rads[1:] - orbital_rads[:-1] #this is dr/dt
pericenters = np.where(np.logical_and(drdt[1:] > 0, drdt[:-1] < 0))[0] + 1
#we verified that adding one to the loc does indeed line up correctly with pericenter
msk = mask_pericenters(test_data)
apo_msk = mask_apocenters(test_data)
fig, ax = plot()
plt.plot(times[msk],orbital_rads[msk])
plt.plot(times,orbital_rads)
plt.plot(times[apo_msk], orbital_rads[apo_msk], '*')
#plt.plot(times,orbital_rads)
for pc in pericenters:
    plt.axvline(times[pc])
#plt.ylim(0,1.2)
#plt.xlim(0,10)

#so now we know the locations of pericenter
#after this, we just need an estimate for the length of the orbital period, so we are removing ~5-10%
#estimate of orbital period is just finding the mean difference between the sign_flips and multiplying it by delta_t


# In[14]:


def load_sim(directory,cs_num,mask_snaps=True,normalize=True,by_nfw=True, mask_type='peris'):
    prof_dat = np.loadtxt(directory / 'radprof_rho.txt')[1:,1:] #radii are in 0th and same for all
    sh_dat = np.loadtxt(directory / 'subhalo_evo.txt')
    num_conv_ind = convergence_criteria(sh_dat,cs_vals[cs_num])
    #if(num_conv_ind >= 0):
    #    prof_dat = prof_dat[:num_conv_ind,:]
    #    sh_dat = sh_dat[:num_conv_ind,:]
    #else:
    #    num_conv_ind = timesteps
    #now, we run the pericenter masking first, then check for convergence
    #this fixes issues such as sim number 2,5,2,0 where the convergence cutoff is near, but before, a pericenter
    #and then a region near the last pericenter doesn't get masked
    
    # by_nfw decides if we normalize by NFW or by the first snapshot, both of which are very similar
    # different than normalizing by isolated halo though, which I don't think we any longer need to do
    
    if(mask_snaps):
        if(mask_type=='peris'):
            mask = mask_pericenters(sh_dat)[0:num_conv_ind]
        elif(mask_type=='apos'):
            mask = mask_apocenters(sh_dat)[0:num_conv_ind]
        else:
            print("Invalid type of mask!")
            return None
        prof_dat = prof_dat[:num_conv_ind,:][mask]
        sh_dat = sh_dat[:num_conv_ind,:][mask]
        if(normalize):
            if(by_nfw):
                return(np.column_stack((sh_dat[:,7],prof_dat/avg_nfw_profs[cs_num,:])))
            else:
                return(np.column_stack((sh_dat[:,7],prof_dat / prof_dat[0,:])))
        else:
            return(np.column_stack((sh_dat[:,7],prof_dat)))
    else:
        prof_dat = prof_dat[:num_conv_ind,:]
        sh_dat = sh_dat[:num_conv_ind,:]
        if(normalize):
            if(by_nfw):
                return(np.column_stack((sh_dat[:,7],prof_dat/avg_nfw_profs[cs_num,:])))
            else:
                return(np.column_stack((sh_dat[:,7],prof_dat / prof_dat[0,:])))
        else:
            return(np.column_stack((sh_dat[:,7],prof_dat)))


# In[15]:


#to-do: make sure that the simulation radii are the same for all sims
#will want to store the relevant simulation parameters for re-scaling? or not necessary
#probably will be necessary for when we want to see where most of the variance comes from
def load_sim_old(directory,cs_num,mask_snaps=True,normalize=True,by_iso=True, mask_type='peris'):
    prof_dat = np.loadtxt(directory / 'radprof_rho.txt')[1:,1:] #radii are in 0th and same for all
    sh_dat = np.loadtxt(directory / 'subhalo_evo.txt')
    num_conv_ind = convergence_criteria(sh_dat,cs_vals[cs_num])
    #if(num_conv_ind >= 0):
    #    prof_dat = prof_dat[:num_conv_ind,:]
    #    sh_dat = sh_dat[:num_conv_ind,:]
    #else:
    #    num_conv_ind = timesteps
    #now, we run the pericenter masking first, then check for convergence
    #this fixes issues such as sim number 2,5,2,0 where the convergence cutoff is near, but before, a pericenter
    #and then a region near the last pericenter doesn't get masked
    
    if(mask_snaps):
        if(mask_type=='peris'):
            mask = mask_pericenters(sh_dat)[0:num_conv_ind]
        elif(mask_type=='apos'):
            mask = mask_apocenters(sh_dat)[0:num_conv_ind]
        else:
            print("Invalid type of mask!")
            return None
        prof_dat = prof_dat[:num_conv_ind,:][mask]
        sh_dat = sh_dat[:num_conv_ind,:][mask]
        if(normalize):
            if(by_iso):
                return(np.column_stack((sh_dat[:,7],prof_dat/isolated_final_profs[cs_num,:num_conv_ind,:][mask])))
            else:
                return(np.column_stack((sh_dat[:,7],prof_dat / prof_dat[0,:])))
        else:
            return(np.column_stack((sh_dat[:,7],prof_dat)))
    else:
        prof_dat = prof_dat[:num_conv_ind,:]
        sh_dat = sh_dat[:num_conv_ind,:]
        if(normalize):
            if(by_iso):
                return(np.column_stack((sh_dat[:,7],prof_dat/isolated_final_profs[cs_num,:num_conv_ind,:])))
            else:
                return(np.column_stack((sh_dat[:,7],prof_dat / prof_dat[0,:])))
        else:
            return(np.column_stack((sh_dat[:,7],prof_dat)))
    #then, call a function that finds regions near pericentric passage and masks those points...
    #idea: find all places where dr/dt changes from negative to positive, estimate the period by how many time
    #steps in between as N_p and then remove 5-10% of points surrounding each of the pericentric passages
        
    #will do a check for numerical stability on the subhalo evolution at each timestep and then return
    #all timesteps that pass the check; will want to make the first column of the matrix the simulation ID
    #so that we can have a second array with the simulation IDs and the corresponding simulation parameters    


# In[16]:


def rmax0(rs): # in units where Rvir=1, rs=1/cs
    return 2.163*rs

def Vmax0(c,Vvir): #in units where Rvir=1=Mvir, you have Vvir=sqrt(1*1/1) = 1.
    return 0.465*Vvir * np.sqrt(c / NFWf(c))


# In[17]:


def load_vrmax(directory, cs_num, normed=True):
    prof_dat = np.loadtxt(directory / 'radprof_m.txt')[1:,1:] #radii are in 0th and same for all
    sh_dat = np.loadtxt(directory / 'subhalo_evo.txt')
    num_conv_ind = convergence_criteria(sh_dat,cs_vals[cs_num])
    
    #now, find the apocenters and remove other points
    mask = mask_apocenters(sh_dat)[0:num_conv_ind]
    prof_dat = prof_dat[:num_conv_ind,:][mask]
    sh_dat = sh_dat[:num_conv_ind,:][mask]

    #compute vmax, rmax for each point
    vels = np.sqrt(G * prof_dat / mass_prof_radii)
    rmax = []
    vmax = []
    for i in range(0,vels.shape[0]):
        interp_vel_prof = InterpolatedUnivariateSpline(mass_prof_radii,vels[i,:], k=4)
        rm = max(interp_vel_prof.derivative().roots())
        rmax.extend([rm])
        vmax.extend([interp_vel_prof(rm)])
        #plt.plot(np.log10(radii), np.log10(vels[2,:]))
        #plt.plot(np.log10(radii), np.log10(interp_vel_prof(radii)))
        #plt.axvline(np.log10(rm)) #for visualizing
             
    if(normed):
        init_vals = np.column_stack((np.repeat(vmax[0],len(vmax)-1),np.repeat(rmax[0],len(rmax)-1)))
        #rmax = np.array(rmax) / rmax[0] #normalized
        #vmax = np.array(vmax) / vmax[0]
        
        rmax = np.array(rmax) / rmax0(1. / cs_vals[cs_num]) # want to try the old version again to figure out issue...
        vmax = np.array(vmax) / Vmax0(cs_vals[cs_num], 1.)  # basically, should we be comparing to NFW or to t=0, why does the difference come about?
    else:
        rmax = np.array(rmax)
        vmax = np.array(vmax)
      
    
    return(np.column_stack((sh_dat[:,7], vmax, rmax, prof_dat))[1:,:],init_vals) #throw out the first one


# In[18]:


print(np.loadtxt(dash_root / ch_dirs[0] / cs_dirs[10] / xc_dirs[5] / eta_dirs[5]/'radprof_rho.txt')[1:,1:].shape)
print(np.loadtxt(dash_root / ch_dirs[0] / cs_dirs[10] / xc_dirs[5] / eta_dirs[5]/'subhalo_evo.txt').shape)

#array will be (301 timesteps * 11**4 sims) by (4sim params + 1f_b + 40 profile values)
#after building matrix, remove all rows with just zeros


# In[19]:


dat_matrix = np.zeros((timesteps*nsims, 45))#[] #preload this as numpy zeros of the proper size
unnormalized_dat_matrix = np.zeros((timesteps*nsims, 41))

sims_present = []

MT = 'peris'
mask_or_not = True

#loop over directories
#for each directory, check if subhalo evo and rho are present
#if they are, load them with load_sim, check which timesteps are valid, return number
#append only the relevant rows of the sim data in a matrix with cols: sim params, f_b, rho vals for each time step
for i in range(0,11):
    print("i",i)
    for j in range(0,11):
        print("j",j)
        for k in range(0,11):
            for l in range(0,11):
                direct = dash_root / ch_dirs[i] / cs_dirs[j] / xc_dirs[k] / eta_dirs[l]
                pf_fn = direct / 'radprof_rho.txt'
                sh_evo_fn = direct / 'subhalo_evo.txt'
                if(Path(pf_fn).is_file()):
                    sims_present.append([i,j,k,l])
                    sim_dat = load_sim(direct,j,mask_snaps=mask_or_not, normalize=True, by_nfw=True, mask_type=MT)#[0,:][np.newaxis,:]
                    snaps_used = sim_dat.shape[0]
                    row = ((i*11**3)+(j*11**2)+(k*11)+l)*timesteps
                    dat_matrix[row:row+snaps_used,0] = ch_vals[i]
                    dat_matrix[row:row+snaps_used,1] = cs_vals[j]
                    dat_matrix[row:row+snaps_used,2] = xc_vals[k]
                    dat_matrix[row:row+snaps_used,3] = eta_vals[l]
                    dat_matrix[row:row+snaps_used,4:] = sim_dat
                    unnormalized_dat_matrix[row:row+snaps_used,:] = load_sim(direct,j,mask_snaps=mask_or_not,normalize=False, mask_type=MT)#[0,:][np.newaxis,:]
                    
                    #dat_matrix.append(load_sim(direct,cs[j])) #going to need to be able to read off the cs value as well
dat_matrix = dat_matrix[~np.all(dat_matrix == 0, axis=1)]

unnormalized_dat_matrix = unnormalized_dat_matrix[~np.all(unnormalized_dat_matrix == 0, axis=1)]


# In[20]:


len(sims_present) #this says that there are 2177 simulations that have run


# In[21]:


sims_present = np.array(sims_present)


# In[22]:


print(np.unique(sims_present[:,0], return_counts=True))
print(np.unique(sims_present[:,1], return_counts=True))
print(np.unique(sims_present[:,2], return_counts=True))
print(np.unique(sims_present[:,3], return_counts=True))


# In[23]:


logfb_plot = np.linspace(-3,0,300)

fig, ax = plot()
n, bins, patches = plt.hist(np.log10(dat_matrix[:,4]),bins=int(np.sqrt(len(dat_matrix))),density='normed',histtype='bar', ec='black');
bin_centers = (bins[1:] + bins[:-1]) * 0.5
weight_func = InterpolatedUnivariateSpline(bin_centers[:], n[:], k=2,ext=3)
plt.plot(logfb_plot,weight_func(logfb_plot))
plt.xlabel('$\log_{10}(f_b)$')
plt.ylabel('N (timesteps across all DASH sims)')
plt.ylim(0,5)
#plt.ylim(-0.1, 0.1)

def fb_weight(fb): #proportional to 1/histogram here
    return 1. / weight_func(np.log10(fb))

#let's fit a kernel density estimation to this so that we can use the inverse of it as a weight to get equal
#value for all logarithmic bins of f_b

#there may be some motivation for logarithmically binning the f_b values? ask Frank...

#not exactly a flat pdf of things to play with, but we can at least look at binned regions


# In[24]:


plt.plot(logfb_plot, np.log10(fb_weight(10**logfb_plot)))
#not as flat as I would like...


# ## Orbital parameter PDF calculations

# In[25]:


#pdfs for the orbital parameters
#see if these can come from something more optimized for numerical speed at some point...
def plor(x,gam):
    return (gam / np.pi)*1.0/(x*x + gam*gam)

def pgauss(x,sig,mu):
    p1 = 1.0 / (np.sqrt(2.*np.pi)*sig)
    p2 = np.exp(-1.*(x-mu)**2 / (2.*sig**2))
    return p1*p2

def pV(x,sig,gam,mu):
        return integrate.quad(lambda xp: pgauss(xp,sig,mu)*plor(x-xp,gam), -1*np.inf, np.inf,full_output=1)[0];

#Go's convolution... we can just use integrate.quad
#def pV(x,sig,gam,mu):
#    p = 0.0;
#    xp = -10.0;
#    dx = 0.01;
#    while(xp <=10.0):
#        p = p + dx*pgauss(xp,sig,mu)*plor(x-xp,gam)
#        xp = xp +  dx;  
#   return p;
    
def pVr(x,B):
    return jiang_A * (np.exp(B*x)-1.)


jiang_sig = 0.077
jiang_mu  = 1.220
jiang_gam = 0.109
jiang_B = 0.049
jiang_A = 1.0 / integrate.quad(lambda xp: np.exp(jiang_B*xp)-1., 0., 1.)[0]
#everything in the pdfs for Vr/V, V/Vvirh look correct, so just need to verify coordinate transformations


# In[26]:


print(eta_vals)
print(xc_vals)


# In[27]:


#translation of Go's code; only thing that disagrees with mine upon visual inspection is the result after
#Jacobian, so our Jacobian was messed up; all good otherwise

#can probably speed it up by reducing calls by doing array-wise calculation of Jacobian instead of each iteration

def transf(eta, xc, ch):
    V_Vvir = np.sqrt((2.*np.log(1+ch) - (2*np.log(1+ch*xc) - NFWf(ch*xc))/xc)/NFWf(ch))
    Vt_Vvir = eta*np.sqrt(xc * NFWf(ch*xc)/NFWf(ch))
    Vr_V = np.sqrt(1. - Vt_Vvir**2 / V_Vvir**2)
    return np.array([Vr_V, V_Vvir])

#this method depends on the bin width, so we need to do some factor of 10 such that we still get all of the
#necessary samples


n_prob_bins = 100
xcs = np.logspace(np.log10(0.5),np.log10(2.0),n_prob_bins+1)
ets = np.linspace(0.0,0.999,n_prob_bins+1)
print(xcs)
print(ets)
vr_vt_mat = np.zeros((len(xcs),len(ets),2)) #eta is x, xc is y
for i in range(0,len(xcs)):
    for j in range(0,len(ets)):
        vr_vt_mat[i,j,:] = transf(ets[j], xcs[i], 5.) #c_h=5
#my transformations were fine, but the jacobian wasn't working properly...

pdf_mat = np.zeros((n_prob_bins,n_prob_bins))
for i in range(0,n_prob_bins):
    for j in range(0,n_prob_bins):
        pdf_mat[i,j] = pVr(vr_vt_mat[i,j,0], jiang_B) * pV(vr_vt_mat[i,j,1],jiang_sig, jiang_gam, jiang_mu)
        if(np.isnan(pdf_mat[i,j])):
            pdf_mat[i,j] = 0
            continue
        else:
            pdf_mat[i,j] *= np.abs(np.sqrt(1-ets[j]**2) - np.sqrt(1-ets[j+1]**2)) * np.abs(np.sqrt(1/xcs[i]) - np.sqrt(1/xcs[i+1]))
                
fig, ax = plot()
plt.plot(ets[:-1],np.nansum(pdf_mat,axis=0)); plt.xlabel(r'$\eta$'); plt.ylabel('pdf')
fig, ax = plot()
plt.plot(xcs[:-1],np.nansum(pdf_mat,axis=1)); plt.xlabel(r'$x_c$'); plt.ylabel('pdf')

#the xc values vary along 0th axis, eta values vary along 1st axis...

fig, ax = plot()
plt.contour(ets[:-1],xcs[:-1],pdf_mat)


# In[28]:


#NFW properties: potential, enclosed mass
#need to solve conservation of energy equation for Vr, Vtheta

def NFWpotential(r, Vvir, Rvir, ch):
    top = -1. * Vvir * Vvir * np.log(1. + ch*r/Rvir)
    bottom = NFWf(ch)*r/Rvir
    return top/bottom

def NFWmass(r,Mvir,Rvir, c):
    return Mvir * NFWf(c*r/Rvir) / NFWf(c)

def NFWVc2(r, Mvir, Rvir, c):
    #squared circular velocity #G=1
    return NFWmass(r, Mvir, Rvir, c) / r

def NFWVc(r, Mvir, Rvir, c):
    #circular velocity #G=1
    return np.sqrt(NFWmass(r, Mvir, Rvir, c) / r)


# ## $\rho(r| f_\mathrm{b}, c_\mathrm{s})$ fitting function optimization

# In[29]:


print(dat_matrix.shape)
print(radii[:30].shape) #lose 10 additional bins...

#we only want to keep the first 30 radial bins, because problems arise outside of the virial radius since we
#are actually calculating bound fraction relative to isolated halo to avoid artificial 2-body relaxation
#so there are 579528 snapshots that are not near pericenter and pass numerical stability test

#dat matrix contains ch, cs, xc, eta, fb, radial pts
innermost_radial_pt = 10
radial_pts =  30 - innermost_radial_pt
cols_needed = np.append(np.array([0,1,2,3,4]),np.arange(5+innermost_radial_pt, 5+innermost_radial_pt+radial_pts))
dat_for_fit = dat_matrix[:,cols_needed]
cols_needed = np.arange(1+innermost_radial_pt, 1+innermost_radial_pt+radial_pts)
unnormed_dat_for_fit = unnormalized_dat_matrix[:,cols_needed] #contains fb still, drop

#########################################
# THIS IS JUST TO TEST A SMALL SUBSET
#unnormed_dat_for_fit = unnormed_dat_for_fit[dat_for_fit[:,0] == ch_vals[2]]
#dat_for_fit = dat_for_fit[dat_for_fit[:,0] == ch_vals[2]]
#unnormed_dat_for_fit = unnormed_dat_for_fit[dat_for_fit[:,1] == cs_vals[5]]
#dat_for_fit = dat_for_fit[dat_for_fit[:,1] == cs_vals[5]]
#unnormed_dat_for_fit = unnormed_dat_for_fit[dat_for_fit[:,2] == xc_vals[6]]
#dat_for_fit = dat_for_fit[dat_for_fit[:,2] == xc_vals[6]]
#unnormed_dat_for_fit = unnormed_dat_for_fit[dat_for_fit[:,3] == eta_vals[7]]
#dat_for_fit = dat_for_fit[dat_for_fit[:,3] == eta_vals[7]]

#####select for only a specific cs value
#csn = 10
#unnormed_dat_for_fit = unnormed_dat_for_fit[dat_for_fit[:,1] == cs_vals[csn]]
#dat_for_fit = dat_for_fit[dat_for_fit[:,1] == cs_vals[csn]]

#### WE SHOULD THROW OUT FB=1, SINCE WE DON'T WANT TO CALIBRATE THAT. MODEL SHOULD BE UNITY AT Fb=1
#unnormed_dat_for_fit = unnormed_dat_for_fit[dat_for_fit[:,4] != 1]
#dat_for_fit = dat_for_fit[dat_for_fit[:,4] != 1]

'''
subset_frac = 0.1
mask = np.full(dat_for_fit.shape[0], False)
mask[:int(dat_for_fit.shape[0] * subset_frac)] = True
np.random.shuffle(mask)
unnormed_dat_for_fit = unnormed_dat_for_fit[mask]
dat_for_fit = dat_for_fit[mask]
'''

########################################



fit_rads = radii[innermost_radial_pt:innermost_radial_pt+radial_pts]

print(dat_for_fit.shape)
print(unnormed_dat_for_fit.shape)
print(fit_rads.shape)
print(fit_rads)


# In[30]:


#make a mask that throws out NaNs, zeros, and infinities
#TODO: figure out where these are coming from
#remember to change this depending on if we're doing the calculation normalized to isolated subhalo
#or normalized to t=0
#msk  = ~np.logical_or(np.any(np.isnan(dat_for_fit[:,5:]),axis=1), 
#                      np.logical_or(np.any(dat_for_fit[:,5:] == 0., axis=1), 
#                                    np.any(dat_for_fit[:,5:] == np.inf, axis=1)))

##### ALLOWING ZEROS INTO DATASET; doesn't affect calculation since error is infinite #####
msk  = ~np.logical_or(np.any(np.isnan(dat_for_fit[:,5:]),axis=1), np.any(dat_for_fit[:,5:] == np.inf, axis=1))


# In[31]:


print(np.sum(np.isnan(dat_for_fit[:,5:]))) #this happens when there was a zero....
#let's just throw out the 1.6k points where there is a nan, because these are just where the denominator was zero
#meaning that there were no points in that cell at a given snapshot in the isolated halo
#dat_for_fit = dat_for_fit[~np.any(np.isnan(dat_for_fit[:,5:]),axis=1)]
#dat_for_fit = dat_for_fit[~np.any(dat_for_fit[:,5:] == 0., axis=1)]
#dat_for_fit = dat_for_fit[~np.any(dat_for_fit[:,5:] == np.inf, axis=1)]
dat_for_fit = dat_for_fit[msk]
unnormed_dat_for_fit = unnormed_dat_for_fit[msk]
print(dat_for_fit.shape)


# In[32]:


#mask bad data points by replacing with zero all radial bins that start to increase again
#there are 30 points, let's remove the inner few radii and the outer ones whenever we have two trues in a row?
#increasing_mask =  dat_for_fit[:,6:] > dat_for_fit[:,5:-1]
#start_col = 15 #starting from the 15th radial bin
#dbl_increasing_mask = increasing_mask[:,start_col:-1] * increasing_mask[:,start_col+1:]
#print(np.sum(increasing_mask[:,start_col:]) / (np.shape(increasing_mask[:,start_col:])[0]*np.shape(increasing_mask[:,start_col:])[1]))
#print(np.sum(dbl_increasing_mask) / (np.shape(dbl_increasing_mask)[0]*np.shape(dbl_increasing_mask)[1]))
#for i in range(0,Nsnaps):
#    for j in range(start_col, n_prof_pts-2):
#        if(dbl_increasing_mask[i,j-start_col]): #if start column meets requirement of 2 radial bins increasing, kill rest of prof
#            dat_for_fit[i,5+j:] = 0
#            break #no need to keep going here, since we kill rest of the bins

#################fix implementation
#################currently removing all bins outside of the inner 15 (which have some fluctuation)
#################that have two radial bins increasing in a row


# In[33]:


#now that we have the pdf for the orbital parameter, let's do the fit.

Nsnaps = dat_for_fit.shape[0]
n_prof_pts = dat_for_fit.shape[1] - 5 #-5 because ch, cs, xc, eta, fb


# In[34]:


sim_inds = np.zeros((Nsnaps,2))
#sim inds is eta first, then xc
for i in range(0,Nsnaps):
    sim_inds[i] = [np.where(dat_for_fit[i,3] == eta_vals)[0][0], np.where(dat_for_fit[i,2] == xc_vals)[0][0]]
    
#since we are using 100 points in our pdf, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99
#just need to make the reduced pdf matrix
sim_inds = sim_inds.astype(int)


# In[35]:


print(ets[0::10])
print(xcs[0::10])
inds_to_take = [0,10,20,30,40,50,60,70,80,90,99]
print(pdf_mat[inds_to_take][:,inds_to_take].shape)
sub_pdf_mat = pdf_mat[inds_to_take][:,inds_to_take]


# In[36]:


weights = np.zeros(Nsnaps)
#just need the probabilities
#for i in range(0,Nsnaps):
    #weights[i] = sub_pdf_mat[sim_inds[i,0],sim_inds[i,1]] * fb_weight(dat_for_fit[i,4]) #matrix is eta by xc
#    weights[i] = fb_weight(dat_for_fit[:,4])
#TODO: Verify that weights are calculated correctly, if the order was flipped the fit would look much worse
#eta by xc vs other way around
weights = fb_weight(dat_for_fit[:,4])
W_tot = np.sum(weights)


# In[37]:


# new weights
weights = np.zeros((Nsnaps, n_prof_pts))
rmax_vals_nfw = 2.163 / cs_vals
weights = 1.0 / (fit_rads[np.newaxis,:] / (2.163 / dat_for_fit[:,1,np.newaxis]))
weights[weights >=1] = 1.


# In[38]:


errors = np.zeros((Nsnaps, n_prof_pts))
error_prefact = (1. / np.log(10)) * (mp / rho200_0)**(1./2.)
for i in range(0,Nsnaps):
    #errors[i,:] = 1.0 / np.sqrt(unnormed_dat_for_fit[i,:] * shell_vols[:n_prof_pts])
    errors[i,:] = error_prefact / np.sqrt(unnormed_dat_for_fit[i,:] * shell_vols[innermost_radial_pt:innermost_radial_pt+n_prof_pts])


# In[39]:


plt.hist(np.log10(errors[errors != np.inf]))
plt.xlabel('log10(error)')
plt.ylabel('N')


# In[40]:


#dat_for_fit[:,5:][dat_for_fit[:,5:] == 0] = np.finfo(np.float).eps #machine epsilon


# In[43]:


#for a given dataset, we have the weights, errors, dat_for_fit (contains params and prof pts normed to iso)
#this is all we need in order to run the following cost function minimization and mcmc, so we can pickle
#it and load it in my codes on grace
np.savez('fit_dat_avgnfw_0.01rv_no1Rv_all_PERIS', dat_for_fit=dat_for_fit, weights=weights, errors=errors, fit_rads=fit_rads)

#norms to try: avg_nfw (best so far, can try removing another data point or varying Vmax0), t=0, iso, true NFW
# want to see which one gets the best Vmax, Rmax distributions for our particular model
# then, can try reducing further in to just focus on the radial region of interest
# let's look at the residual distribution after this...

# we saw that the rmax0 / rmax_NFW is slightly skewed to less than 1
# so if we want to get the predictions closer to correct, we should normalize by Rmax0

# we're trying with the outer two radial bins removed; Depending on how this looks, we may keep it
# otherwise, I think that we need to start out by trying to increase parametrization and c_s dependence on f_t, because
# there is a clear inner normalization residual with f_te


# In[41]:


def plot_rho_vs_model(ch_num, cs_num, xc_num, eta_num, snapnum, fp, model_param_func, rho_model,by_iso=False):
    direct = dash_root / ch_dirs[ch_num] / cs_dirs[cs_num] / xc_dirs[xc_num] / eta_dirs[eta_num]
    pf_fn = direct / 'radprof_rho.txt'
    sh_evo_fn = direct / 'subhalo_evo.txt'
    if(Path(pf_fn).is_file()):
        sim_dat = load_sim_old(direct,cs_num,False,normalize=True, by_iso=by_iso) #keep all timesteps
        
        #generate model, takes as input r/rs0
        rho_m = rho_fit(radii * cs_vals[cs_num], np.array([sim_dat[snapnum,0]]), np.array([cs_vals[cs_num]]), fp, model_param_func, rho_model)
        fig, ax = loglogplot()
        plt.xlabel(r'$r/r_\textrm{s,0}$')
        plt.ylabel(r'$\rho(r) / \rho(r,\textrm{iso})$')
        plt.plot(radii * cs_vals[cs_num], sim_dat[snapnum,1:], label='DASH')
        plt.plot(radii * cs_vals[cs_num], rho_m, label='Model')
        plt.legend()
        plt.title('$c_h = %.2f$, $c_s = %.2f$, $x_c = %.2f$, $\eta = %.2f$, $f_b = %.2f$' % (ch_vals[ch_num], cs_vals[cs_num], xc_vals[xc_num], eta_vals[eta_num], sim_dat[snapnum,0]))
        plt.xlim(10**-2 * cs_vals[cs_num], 1.* cs_vals[cs_num])
        plt.ylim(10**-4, 2.)
        
    else:
        print("No simulation run here yet!")


# In[42]:


# one last try with njit for use on the clusters...

# TODO: eventually incorporate c_h?

o_errsq = 1. / errors**2
o_err = 1. / errors

#weight_o_errsq = weights[:, np.newaxis] * o_errsq
#weights_no_err = np.repeat(weights, len(fit_rads)).reshape(np.shape(o_errsq))
#weights_no_err[dat_for_fit[:, 5:] == 0] = 0
unity = np.ones(np.shape(o_errsq))
unity[dat_for_fit[:, 5:] == 0] = 0
r_by_rs = fit_rads*dat_for_fit[:, 1][:, np.newaxis]
fb_cs = np.column_stack(
    (dat_for_fit[:, 4], dat_for_fit[:, 1]))  # fb, cs for use_cs

ft_vals = 10**(0.08574221*(fb_cs[:, 1] / 10.)**-0.35442253 * np.log10(fb_cs[:, 0]
                                                                      ) + -0.09686971*(fb_cs[:, 1] / 10.)**0.41424968 * np.log10(fb_cs[:, 0])**2)

MIN_PROF_DENS = 1e-12
PRIOR_MAX = 10.


@njit
def rho_model_hayashi(r, mp):
    return mp[0] / (1. + (r/mp[1])**mp[2])


@njit
def exp_decay_model(r, mp):
    return mp[0] * np.exp(-1.*r / mp[1])


@njit
def exp_decay_deluxe(r, mp):
    return mp[0] * np.exp(-1. * (r / mp[1])**mp[2])


@njit
def exp_decay_v3(r, mp):
    # f_t, c_s, and r_t
    return mp[0] * np.exp(-1. * r * ((mp[1] - mp[2])/(mp[1]*mp[2])))


@njit
def hayashi_deluxe(r, mp):
    # f_t, r_vir (i.e., c_s), r_t, delta
    if(np.abs(mp[1]-mp[2]) < 1e-8):
        #print("this happened for r=%.3e"%r)
        return mp[0]
    elif(mp[1] < mp[2]): #rt is larger than rvir, which shouldn't be the case...
        return 0. #this will throw an inf
    else:
        return mp[0] / (1. + (r * ((mp[1] - mp[2])/(mp[1]*mp[2])))**mp[3])

@njit
def hayashi_deluxe_deluxe(r, mp):
    # f_t, r_vir (i.e., c_s), r_t, delta
    if(np.abs(mp[1]-mp[2]) < 1e-8):
        return mp[0]
    elif(mp[1] < mp[2]): #rt is larger than rvir, which shouldn't be the case...
        return 0. #this will throw an inf
    else:
        return mp[0] / (1. + (r * ((mp[1] - mp[2])/(mp[1]*mp[2])))**mp[3])**mp[4]
    
    
# will try this if our double power-law can't cut it
@njit
def powerlaw_exp(r, mp):
    if(np.abs(mp[1]-mp[2]) < 1e-8):
        #print("this happened for r=%.3e"%r)
        return mp[0]
    elif(mp[1] < mp[2]): #rt is larger than rvir, which shouldn't be the case...
        return 0. #this will throw an inf
    else:
        return (mp[0] / (1. + (r * ((mp[1] - mp[2])/(mp[1]*mp[2])))**mp[3])) * np.exp(-1. * r * ((mp[1] - mp[4])/(mp[1]*mp[4])))
   
    
@njit(parallel=True, fastmath=True)
def paramet_plexp_v3(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        model_params[i, 0] = 10**(0. + fp[0] * (fb_cs[i,1] / 10.)**fp[1] * np.log10(fb_cs[i,0]) + fp[2] * (1. - fb_cs[i,0])**fp[3] * np.log10(fb_cs[i,1]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * np.log10(fb_cs[i,0]) + fp[6] * (1. - fb_cs[i,0])**fp[7] * np.log10(fb_cs[i,1])) * np.exp(fp[8] * (fb_cs[i,1] / 10.)**fp[9] * (1. - fb_cs[i,0]))
        model_params[i, 3] = 10**(fp[10] + fp[11]*(fb_cs[i,1] / 10.)**fp[12] * np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(np.log10(fb_cs[i,1]) + fp[13] * (fb_cs[i,1] / 10.)**fp[14] * np.log10(fb_cs[i,0]) + fp[15] * (fb_cs[i,1] / 10.)**fp[16] * np.log10(fb_cs[i,0])**2)
    return model_params    

@njit(parallel=True, fastmath=True)
def paramet_plexp_v2(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        model_params[i, 0] = 10**(0. + fp[0] * (fb_cs[i,1] / 10.)**fp[1] * np.log10(fb_cs[i,0]) + fp[2] * (1. - fb_cs[i,0])**fp[3] * np.log10(fb_cs[i,1]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * np.log10(fb_cs[i,0]) + fp[6] * (1. - fb_cs[i,0])**fp[7] * np.log10(fb_cs[i,1])) * np.exp(fp[8] * (fb_cs[i,1] / 10.)**fp[9] * (1. - fb_cs[i,0]))
        model_params[i, 3] = 10**(fp[10] + fp[11]*(fb_cs[i,1] / 10.)**fp[12] * np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(np.log10(fb_cs[i,1]) + fp[13] * (fb_cs[i,1] / 10.)**fp[14] * np.log10(fb_cs[i,0]))
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_plexp_v1(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        model_params[i, 0] = 10**(0. + fp[0] * (fb_cs[i,1] / 10.)**fp[1] * np.log10(fb_cs[i,0]) + fp[2] * (1. - fb_cs[i,0])**fp[3] * np.log10(fb_cs[i,1]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * np.log10(fb_cs[i,0]) + fp[6] * (1. - fb_cs[i,0])**fp[7] * np.log10(fb_cs[i,1])) * np.exp(fp[8] * (fb_cs[i,1] / 10.)**fp[9] * (1. - fb_cs[i,0]))
        model_params[i, 3] = 10**(fp[10] + fp[11]*(fb_cs[i,1] / 10.)**fp[12] * np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(np.log10(fb_cs[i,1]) + fp[13] * (fb_cs[i,1] / 10.)**fp[14] * np.log10(fb_cs[i,0]) + fp[15] * (1. - fb_cs[i,0])**fp[16] * np.log10(fb_cs[i,1]))
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_polexp1(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5)) # there's ft, cs, rt, mu, delta
    x = np.log10(fb_cs[:,0])
    y = np.log10(fb_cs[:,1])
    for i in prange(0, len(fb_cs)):
        model_params[i, 0] = 10**(fp[0]*x[i] + fp[1]*y[i] + fp[2]*x[i]*y[i] + fp[3]*x[i]**2 + fp[4]*y[i]**2 + fp[5]*x[i]**2 * y[i] + fp[6]*x[i]*y[i]**2 + fp[7]*x[i]**3 + fp[8]*y[i]**3)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(y[i] + fp[9]*x[i] + fp[10]*y[i] + fp[11]*x[i]*y[i] + fp[12]*x[i]**2 + fp[13]*y[i]**2 + fp[14]*x[i]**2 * y[i] + fp[15]*x[i]*y[i]**2 + fp[16]*x[i]**3 + fp[17]*y[i]**3)
        model_params[i, 3] = 10**(fp[18] + fp[19]*x[i] + fp[20]*y[i] + fp[21]*x[i]**2 + fp[22]*y[i]**2 + fp[23]*x[i]*y[i])
        model_params[i, 4] = 10**(fp[24] + fp[25]*x[i] + fp[26]*y[i] + fp[27]*x[i]**2 + fp[28]*y[i]**2 + fp[29]*x[i]*y[i])
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v46(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # CAN ALSO TRY THIS BUT REPLACING 1-FB WITH LOG(FB)
        model_params[i, 0] = 10**(0. + fp[0] * (fb_cs[i,1] / 10.)**fp[1] * np.log10(fb_cs[i,0]) + fp[2] * (1. - fb_cs[i,0])**fp[3] * np.log10(fb_cs[i,1])) * np.exp(fp[16] * (fb_cs[i,1] / 10.)**fp[17] * (1. - fb_cs[i,0]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * np.log10(fb_cs[i,0]) + fp[6] * (1. - fb_cs[i,0])**fp[7] * np.log10(fb_cs[i,1])) * np.exp(fp[8] * (fb_cs[i,1] / 10.)**fp[9] * (1. - fb_cs[i,0]))
        model_params[i, 3] = 10**(fp[10] + fp[11]*(fb_cs[i,1] / 10.)**fp[12] * np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(fp[13] + fp[14]*(fb_cs[i,1] / 10.)**fp[15] * np.log10(fb_cs[i,0]))
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v51(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # CAN ALSO TRY THIS BUT REPLACING 1-FB WITH LOG(FB)
        model_params[i, 0] = 10**(0. + fp[0] * (fb_cs[i,1] / 10.)**fp[1] * np.log10(fb_cs[i,0]) + fp[2] * (1. - fb_cs[i,0])**fp[3] * np.log10(fb_cs[i,1]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * np.log10(fb_cs[i,0]) + fp[6] * (1. - fb_cs[i,0])**fp[7] * np.log10(fb_cs[i,1])) * np.exp(fp[8] * (fb_cs[i,1] / 10.)**fp[9] * (1. - fb_cs[i,0]))
        model_params[i, 3] = 10**(fp[10] + fp[11] * np.log10(fb_cs[i,1]))
        model_params[i, 4] = 10**(fp[12] + fp[13]*(fb_cs[i,1] / 10.)**fp[14] * np.log10(fb_cs[i,0]) + fp[15] * (1. - fb_cs[i,0])**fp[16] * np.log10(fb_cs[i,1]))
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v53(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # CAN ALSO TRY THIS BUT REPLACING 1-FB WITH LOG(FB)
        model_params[i, 0] = 10**(0. + fp[0] * np.log10(fb_cs[i,0]) + fp[1] * (1. - fb_cs[i,0])**fp[2] * np.log10(fb_cs[i,1]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[3] * (fb_cs[i,1] / 10.)**fp[4] * np.log10(fb_cs[i,0]) + fp[5] * (1. - fb_cs[i,0])**fp[6] * np.log10(fb_cs[i,1])) * np.exp(fp[7] * (fb_cs[i,1] / 10.)**fp[8] * (1. - fb_cs[i,0]))
        model_params[i, 3] = 10**(fp[9] + fp[10]*(fb_cs[i,1] / 10.)**fp[11] * np.log10(fb_cs[i,0]) + fp[12] * (1. - fb_cs[i,0])**fp[13] * np.log10(fb_cs[i,1]))
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v52(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # CAN ALSO TRY THIS BUT REPLACING 1-FB WITH LOG(FB)
        model_params[i, 0] = 10**(0. + fp[0] * (fb_cs[i,1] / 10.)**fp[1] * np.log10(fb_cs[i,0]) + fp[2] * (1. - fb_cs[i,0])**fp[3] * np.log10(fb_cs[i,1]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * np.log10(fb_cs[i,0]) + fp[6] * (1. - fb_cs[i,0])**fp[7] * np.log10(fb_cs[i,1])) * np.exp(fp[8] * (fb_cs[i,1] / 10.)**fp[9] * (1. - fb_cs[i,0]))
        model_params[i, 3] = 10**(fp[10] + fp[11]*(fb_cs[i,1] / 10.)**fp[12] * np.log10(fb_cs[i,0]) + fp[13] * (1. - fb_cs[i,0])**fp[14] * np.log10(fb_cs[i,1]))
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v50(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # CAN ALSO TRY THIS BUT REPLACING 1-FB WITH LOG(FB)
        model_params[i, 0] = 10**(0. + fp[0] * (fb_cs[i,1] / 10.)**fp[1] * np.log10(fb_cs[i,0]) + fp[2] * (1. - fb_cs[i,0])**fp[3] * np.log10(fb_cs[i,1]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * np.log10(fb_cs[i,0]) + fp[6] * (1. - fb_cs[i,0])**fp[7] * np.log10(fb_cs[i,1])) * np.exp(fp[8] * (fb_cs[i,1] / 10.)**fp[9] * (1. - fb_cs[i,0]))
        model_params[i, 3] = 10**(fp[10] + fp[11]*(fb_cs[i,1] / 10.)**fp[12] * np.log10(fb_cs[i,0]) + fp[16] * np.log10(fb_cs[i,1]))
        model_params[i, 4] = 10**(fp[13] + fp[14]*(fb_cs[i,1] / 10.)**fp[15] * np.log10(fb_cs[i,0]) + fp[17] * (1. - fb_cs[i,0])**fp[18] * np.log10(fb_cs[i,1]))
    return model_params

#polynomial expansion once again, but with sufficient dependence baked into mu, delta
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v49(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2. + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(fb_cs[i, 0])**3.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(
            fb_cs[i, 0]) + fp[8]*((fb_cs[i, 1] / 10.)**fp[9]) * np.log10(fb_cs[i, 0])**2. + fp[10]*(fb_cs[i, 1] / 10.)**fp[11] * np.log10(fb_cs[i, 0])**3 + fp[12]*(fb_cs[i, 1] / 10.)**fp[13] * np.log10(fb_cs[i, 0])**4)
        model_params[i, 3] = 10**(fp[14] + fp[15]*(fb_cs[i,1] / 10.)**fp[16] * np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(fp[17] + fp[18]*(fb_cs[i,1] / 10.)**fp[19] * np.log10(fb_cs[i,0]))
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v48(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # CAN ALSO TRY THIS BUT REPLACING 1-FB WITH LOG(FB)
        model_params[i, 0] = 10**(0. + fp[0] * (fb_cs[i,1] / 10.)**fp[1] * np.log10(fb_cs[i,0]) + fp[2] * (1. - fb_cs[i,0])**fp[3] * np.log10(fb_cs[i,1]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * np.log10(fb_cs[i,0]) + fp[6] * (1. - fb_cs[i,0])**fp[7] * np.log10(fb_cs[i,1])) * np.exp(fp[8] * (fb_cs[i,1] / 10.)**fp[9] * (1. - fb_cs[i,0]))
        model_params[i, 3] = 10**(fp[10] + fp[11]*(fb_cs[i,1] / 10.)**fp[12] * np.log10(fb_cs[i,0]) + fp[16] * (1. - fb_cs[i,0])**fp[17] * np.log10(fb_cs[i,1]))
        model_params[i, 4] = 10**(fp[13] + fp[14]*(fb_cs[i,1] / 10.)**fp[15] * np.log10(fb_cs[i,0]) + fp[18] * (1. - fb_cs[i,0])**fp[19] * np.log10(fb_cs[i,1]))
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v47(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # CAN ALSO TRY THIS BUT REPLACING 1-FB WITH LOG(FB)
        model_params[i, 0] = 10**(0. + fp[0] * (fb_cs[i,1] / 10.)**fp[1] * np.log10(fb_cs[i,0]) + fp[2] * (1. - fb_cs[i,0])**fp[3] * np.log10(fb_cs[i,1])) * np.exp(fp[16] * (fb_cs[i,1] / 10.)**fp[17] * (1. - fb_cs[i,0])**fp[18])
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * np.log10(fb_cs[i,0]) + fp[6] * (1. - fb_cs[i,0])**fp[7] * np.log10(fb_cs[i,1])) * np.exp(fp[8] * (fb_cs[i,1] / 10.)**fp[9] * (1. - fb_cs[i,0]))
        model_params[i, 3] = 10**(fp[10] + fp[11]*(fb_cs[i,1] / 10.)**fp[12] * np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(fp[13] + fp[14]*(fb_cs[i,1] / 10.)**fp[15] * np.log10(fb_cs[i,0]))
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v46(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # CAN ALSO TRY THIS BUT REPLACING 1-FB WITH LOG(FB)
        model_params[i, 0] = 10**(0. + fp[0] * (fb_cs[i,1] / 10.)**fp[1] * np.log10(fb_cs[i,0]) + fp[2] * (1. - fb_cs[i,0])**fp[3] * np.log10(fb_cs[i,1])) * np.exp(fp[16] * (fb_cs[i,1] / 10.)**fp[17] * (1. - fb_cs[i,0]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * np.log10(fb_cs[i,0]) + fp[6] * (1. - fb_cs[i,0])**fp[7] * np.log10(fb_cs[i,1])) * np.exp(fp[8] * (fb_cs[i,1] / 10.)**fp[9] * (1. - fb_cs[i,0]))
        model_params[i, 3] = 10**(fp[10] + fp[11]*(fb_cs[i,1] / 10.)**fp[12] * np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(fp[13] + fp[14]*(fb_cs[i,1] / 10.)**fp[15] * np.log10(fb_cs[i,0]))
    return model_params

# this one adds exponential to fte, still has exp in rte but adds power law dpeendence to the 1-fb in the exps for both
# this is also second order in fb for fte, not sure if necessary but we'll decide if it helps based on v44 results
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v45(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    if(fp[3] < 0 or fp[7] < 0 or fp[18] < 0 or fp[19] < 0):
        model_params[:,1] = -100
        return model_params
    for i in prange(0, len(fb_cs)):
        # CAN ALSO TRY THIS BUT REPLACING 1-FB WITH LOG(FB)
        model_params[i, 0] = 10**(0. + fp[0] * (fb_cs[i,1] / 10.)**fp[1] * np.log10(fb_cs[i,0]) + fp[2] * (1. - fb_cs[i,0])**fp[3] * np.log10(fb_cs[i,1])) * np.exp(fp[16] * (fb_cs[i,1] / 10.)**fp[17] * (1. - fb_cs[i,0])**fp[18])
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * np.log10(fb_cs[i,0]) + fp[6] * (1. - fb_cs[i,0])**fp[7] * np.log10(fb_cs[i,1])) * np.exp(fp[8] * (fb_cs[i,1] / 10.)**fp[9] * (1. - fb_cs[i,0])**fp[19])
        model_params[i, 3] = 10**(fp[10] + fp[11]*(fb_cs[i,1] / 10.)**fp[12] * np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(fp[13] + fp[14]*(fb_cs[i,1] / 10.)**fp[15] * np.log10(fb_cs[i,0]))
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v44(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # CAN ALSO TRY THIS BUT REPLACING 1-FB WITH LOG(FB)
        model_params[i, 0] = 10**(0. + fp[0] * (fb_cs[i,1] / 10.)**fp[1] * np.log10(fb_cs[i,0]) + fp[2] * (1. - fb_cs[i,0])**fp[3] * np.log10(fb_cs[i,1]) + fp[16]*(fb_cs[i,1] / 10.)**fp[17] * np.log10(fb_cs[i,0])**2)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * np.log10(fb_cs[i,0]) + fp[6] * (1. - fb_cs[i,0])**fp[7] * np.log10(fb_cs[i,1])) * np.exp(fp[8] * (fb_cs[i,1] / 10.)**fp[9] * (1. - fb_cs[i,0]))
        model_params[i, 3] = 10**(fp[10] + fp[11]*(fb_cs[i,1] / 10.)**fp[12] * np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(fp[13] + fp[14]*(fb_cs[i,1] / 10.)**fp[15] * np.log10(fb_cs[i,0]))
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v43(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # CAN ALSO TRY THIS BUT REPLACING 1-FB WITH LOG(FB)
        model_params[i, 0] = 10**(0. + fp[0] * (fb_cs[i,1] / 10.)**fp[1] * np.log10(fb_cs[i,0]) + fp[2] * (1. - fb_cs[i,0])**fp[3] * np.log10(fb_cs[i,1]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * np.log10(fb_cs[i,0]) + fp[6] * (1. - fb_cs[i,0])**fp[7] * np.log10(fb_cs[i,1])) * np.exp(fp[8] * (fb_cs[i,1] / 10.)**fp[9] * (1. - fb_cs[i,0]))
        model_params[i, 3] = 10**(fp[10] + fp[11]*(fb_cs[i,1] / 10.)**fp[12] * np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(fp[13] + fp[14]*(fb_cs[i,1] / 10.)**fp[15] * np.log10(fb_cs[i,0]))
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v42(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # CAN ALSO TRY THIS BUT REPLACING 1-FB WITH LOG(FB)
        model_params[i, 0] = 10**(0. + fp[0] * (1. - fb_cs[i,0])**fp[1] * np.log10(fb_cs[i,1]) + fp[2] * (fb_cs[i,0] / 10.)**fp[3] * (1. - fb_cs[i,0]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * np.log10(fb_cs[i,0]) + fp[6] * (1. - fb_cs[i,0])**fp[7] * np.log10(fb_cs[i,1]) + fp[8] * (1. - fb_cs[i,0]))
        model_params[i, 3] = 10**(fp[9] + fp[10] * np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(fp[11])
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v40(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # CAN ALSO TRY THIS BUT REPLACING 1-FB WITH LOG(FB)
        model_params[i, 0] = 10**(0. + fp[0] * (fb_cs[i,1] / 10.)**fp[1] * np.log10(fb_cs[i,0]) + fp[2] * (1. - fb_cs[i,0])**fp[3] * np.log10(fb_cs[i,1]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * np.log10(fb_cs[i,0]) + fp[6] * (1. - fb_cs[i,0])**fp[7] * np.log10(fb_cs[i,1]) + fp[8] * (fb_cs[i,1] / 10.)**fp[9] * (1. - fb_cs[i,0]))
        model_params[i, 3] = 10**(fp[10] + fp[11]*(fb_cs[i,1] / 10.)**fp[12] * np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(fp[13] + fp[14]*(fb_cs[i,1] / 10.)**fp[15] * np.log10(fb_cs[i,0]))
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v39(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # CAN ALSO TRY THIS BUT REPLACING 1-FB WITH LOG(FB)
        model_params[i, 0] = 10**(0. + fp[0] * (fb_cs[i,1] / 10.)**fp[1] * np.log10(fb_cs[i,0]) + fp[2] * (1. - fb_cs[i,0])**fp[3] * np.log10(fb_cs[i,1]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * (1. - fb_cs[i,0]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[6] * (fb_cs[i,1] / 10.)**fp[7] * np.log10(fb_cs[i,0]) + fp[8] * (1. - fb_cs[i,0])**fp[9] * np.log10(fb_cs[i,1]) + fp[10] * (fb_cs[i,1] / 10.)**fp[11] * (1. - fb_cs[i,0]))
        model_params[i, 3] = 10**(fp[12] + fp[13]*(fb_cs[i,1] / 10.)**fp[14] * np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(fp[15] + fp[16]*(fb_cs[i,1] / 10.)**fp[17] * np.log10(fb_cs[i,0]))
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v36(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        loggamma1 = np.log10(1. / fp[0]) * fp[1] * (fb_cs[i,1] / 10.)**fp[2] + np.log10(1. + 1./fp[0]) * fp[3] * (fb_cs[i,1] / 10.)**fp[4]
        model_params[i, 0] = 10**(0. + loggamma1 - fp[1] * (fb_cs[i,1] / 10.)**fp[2] * np.log10(fb_cs[i,0] / fp[0]) - fp[3] * (fb_cs[i,1] / 10.)**fp[4] * np.log10(1. + fb_cs[i,0] / fp[0]) + fp[15]*(1. - fb_cs[i,0])**fp[16] * np.log10(fb_cs[i,1]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[5] * (fb_cs[i,1] / 10.)**fp[6] * np.log10(fb_cs[i,0]) + fp[7]*(1. - fb_cs[i,0])**fp[8] * np.log10(fb_cs[i,1]))
        model_params[i, 3] = 10**(fp[9] + fp[10]*(fb_cs[i,1] / 10.)**fp[11] * np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(fp[12] + fp[13]*(fb_cs[i,1] / 10.)**fp[14] * np.log10(fb_cs[i,0]))
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v34(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        model_params[i, 0] = 10**(0. + fp[0] * (fb_cs[i,1] / 10.)**fp[1] * np.log10(fb_cs[i,0]) + fp[2]*(1. - fb_cs[i,0])**fp[3] * np.log10(fb_cs[i,1]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * np.log10(fb_cs[i,0]) + fp[6]*(1. - fb_cs[i,0])**fp[7] * np.log10(fb_cs[i,1]))
        model_params[i, 3] = 10**(fp[8] + fp[9]*(fb_cs[i,1] / 10.)**fp[10] * np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(fp[11] + fp[12]*(fb_cs[i,1] / 10.)**fp[13] * np.log10(fb_cs[i,0]))
    return model_params
    
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v32(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        loggamma1 = np.log10(1. / fp[0]) * fp[1] * (fb_cs[i,1] / 10.)**fp[2] + np.log10(1. + 1./fp[0]) * fp[3] * (fb_cs[i,1] / 10.)**fp[4]
        model_params[i, 0] = 10**(0. + loggamma1 - fp[1] * (fb_cs[i,1] / 10.)**fp[2] * np.log10(fb_cs[i,0] / fp[0]) - fp[3] * (fb_cs[i,1] / 10.)**fp[4] * np.log10(1. + fb_cs[i,0] / fp[0]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        loggamma2 = np.log10(1. / fp[5]) * fp[6] * (fb_cs[i,1] / 10.)**fp[7] + np.log10(1. + 1./fp[5]) * fp[8] * (fb_cs[i,1] / 10.)**fp[9]
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + loggamma2 - fp[6] * (fb_cs[i,1] / 10.)**fp[7] * np.log10(fb_cs[i,0] / fp[5]) - fp[8] * (fb_cs[i,1] / 10.)**fp[9] * np.log10(1. + fb_cs[i,0] / fp[5]) + fp[10]*(1. - fb_cs[i,0])**fp[11] * np.log10(fb_cs[i,1]))
        model_params[i, 3] = 10**(fp[12] + fp[13]*(fb_cs[i,1] / 10.)**fp[14] * np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(fp[15] + fp[16]*(fb_cs[i,1] / 10.)**fp[17] * np.log10(fb_cs[i,0]))
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v31(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        loggamma1 = np.log10(1. / fp[0]) * fp[1] * (fb_cs[i,1] / 10.)**fp[2] + np.log10(1. + 1./fp[0]) * fp[3] * (fb_cs[i,1] / 10.)**fp[4]
        model_params[i, 0] = 10**(0. + loggamma1 - fp[1] * (fb_cs[i,1] / 10.)**fp[2] * np.log10(fb_cs[i,0] / fp[0]) - fp[3] * (fb_cs[i,1] / 10.)**fp[4] * np.log10(1. + fb_cs[i,0] / fp[0]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        loggamma2 = np.log10(1. / fp[5]) * fp[6] * (fb_cs[i,1] / 10.)**fp[7] + np.log10(1. + 1./fp[5]) * fp[8] * (fb_cs[i,1] / 10.)**fp[9]
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + loggamma2 - fp[6] * (fb_cs[i,1] / 10.)**fp[7] * np.log10(fb_cs[i,0] / fp[5]) - fp[8] * (fb_cs[i,1] / 10.)**fp[9] * np.log10(1. + fb_cs[i,0] / fp[5]))
        model_params[i, 3] = 10**(fp[10] + fp[11]*(fb_cs[i,1] / 10.)**fp[12] * np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(fp[13] + fp[14]*(fb_cs[i,1] / 10.)**fp[15] * np.log10(fb_cs[i,0]))
    return model_params
    
#try a polynomial expansion once again, and try adding c_s dep to the slopes?
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v30(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2. + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(fb_cs[i, 0])**3.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(
            fb_cs[i, 0]) + fp[8]*((fb_cs[i, 1] / 10.)**fp[9]) * np.log10(fb_cs[i, 0])**2. + fp[10]*(fb_cs[i, 1] / 10.)**fp[11] * np.log10(fb_cs[i, 0])**3 + fp[12]*(fb_cs[i, 1] / 10.)**fp[13] * np.log10(fb_cs[i, 0])**4)
        model_params[i, 3] = 10**(fp[14] + fp[15]*np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(fp[16] + fp[17]*np.log10(fb_cs[i,0]))
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v29(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        loggamma1 = np.log10(1. / fp[0]) * fp[1] * (fb_cs[i,1] / 10.)**fp[2] + np.log10(1. + 1./fp[0]) * fp[3] * (fb_cs[i,1] / 10.)**fp[4]
        model_params[i, 0] = 10**(0. + loggamma1 - fp[1] * (fb_cs[i,1] / 10.)**fp[2] * np.log10(fb_cs[i,0] / fp[0]) - fp[3] * (fb_cs[i,1] / 10.)**fp[4] * np.log10(1. + fb_cs[i,0] / fp[0]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        loggamma2 = np.log10(1. / fp[5]) * fp[6] * (fb_cs[i,1] / 10.)**fp[7] + np.log10(1. + 1./fp[5]) * fp[8] * (fb_cs[i,1] / 10.)**fp[9]
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + loggamma2 - fp[6] * (fb_cs[i,1] / 10.)**fp[7] * np.log10(fb_cs[i,0] / fp[5]) - fp[8] * (fb_cs[i,1] / 10.)**fp[9] * np.log10(1. + fb_cs[i,0] / fp[5]))
        model_params[i, 3] = 10**(fp[10] + fp[11]*np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(fp[12] + fp[13]*np.log10(fb_cs[i,0]))
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v28_cs(fb_cs, fp): #coupled with hayashi_deluxe_deluxe
    model_params = np.zeros((len(fb_cs), 5))
    for i in prange(0, len(fb_cs)):
        
        model_params[i, 0] = 10**(0. + fp[0]*np.log10(fb_cs[i, 0]) + fp[1]*np.log10(fb_cs[i, 0])**2. + fp[2]*np.log10(fb_cs[i, 0])**3.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[3]*np.log10(fb_cs[i, 0]) + fp[4]*np.log10(fb_cs[i, 0])**2. + fp[5]*np.log10(fb_cs[i, 0])**3 + fp[6]*np.log10(fb_cs[i, 0])**4)
        model_params[i, 3] = 10**(fp[7] + fp[8]*np.log10(fb_cs[i,0]))
        model_params[i, 4] = 10**(fp[9] + fp[10]*np.log10(fb_cs[i,0]))
    return model_params

    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v27_cs(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        
        model_params[i, 0] = 10**(0. + fp[0]*np.log10(fb_cs[i, 0]) + fp[1]*np.log10(fb_cs[i, 0])**2. + fp[2]*np.log10(fb_cs[i, 0])**3.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[3]*np.log10(fb_cs[i, 0]) + fp[4]*np.log10(fb_cs[i, 0])**2. + fp[5]*np.log10(fb_cs[i, 0])**3 + fp[6]*np.log10(fb_cs[i, 0])**4)
        model_params[i, 3] = 10**(fp[7] + fp[8]*np.log10(fb_cs[i,0]))
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v26_cs(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        loggamma1 = np.log10(1. / fp[0]) * fp[1] + np.log10(1. + 1./fp[0]) * fp[2]
        model_params[i, 0] = 10**(0. + loggamma1 - fp[1] * np.log10(fb_cs[i,0] / fp[0]) - fp[2] * np.log10(1. + fb_cs[i,0] / fp[0]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        loggamma2 = np.log10(1. / fp[3]) * fp[4] + np.log10(1. + 1./fp[3]) * fp[5]
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + loggamma2 - fp[4] * np.log10(fb_cs[i,0] / fp[3]) - fp[5] * np.log10(1. + fb_cs[i,0] / fp[3]))
        model_params[i, 3] = 10**(fp[6] + fp[7]*np.log10(fb_cs[i,0]))
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v25(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        loggamma1 = np.log10(1. / fp[0]) * fp[1] * (fb_cs[i,1] / 10.)**fp[2] + np.log10(1. + 1./fp[0]) * fp[3] * (fb_cs[i,1] / 10.)**fp[4]
        model_params[i, 0] = 10**(0. + loggamma1 - fp[1] * (fb_cs[i,1] / 10.)**fp[2] * np.log10(fb_cs[i,0] / fp[0]) - fp[3] * (fb_cs[i,1] / 10.)**fp[4] * np.log10(1. + fb_cs[i,0] / fp[0]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        loggamma2 = np.log10(1. / fp[5]) * fp[6] * (fb_cs[i,1] / 10.)**fp[7] + np.log10(1. + 1./fp[5]) * fp[8] * (fb_cs[i,1] / 10.)**fp[9]
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + loggamma2 - fp[6] * (fb_cs[i,1] / 10.)**fp[7] * np.log10(fb_cs[i,0] / fp[5]) - fp[8] * (fb_cs[i,1] / 10.)**fp[9] * np.log10(1. + fb_cs[i,0] / fp[5]) + fp[12]*(1. - fb_cs[i,0])**fp[13] * np.log10(fb_cs[i,1]))
        model_params[i, 3] = 10**(fp[10] + fp[11]*np.log10(fb_cs[i,0]))
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v24(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        loggamma1 = np.log10(1. / fp[0]) * fp[1] * (fb_cs[i,1] / 10.)**fp[2] + np.log10(1. + 1./fp[0]) * fp[3] * (fb_cs[i,1] / 10.)**fp[4]
        model_params[i, 0] = 10**(0. + loggamma1 - fp[1] * (fb_cs[i,1] / 10.)**fp[2] * np.log10(fb_cs[i,0] / fp[0]) - fp[3] * (fb_cs[i,1] / 10.)**fp[4] * np.log10(1. + fb_cs[i,0] / fp[0]) + fp[12]*(1. - fb_cs[i,0])**fp[13] * np.log10(fb_cs[i,1]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        loggamma2 = np.log10(1. / fp[5]) * fp[6] * (fb_cs[i,1] / 10.)**fp[7] + np.log10(1. + 1./fp[5]) * fp[8] * (fb_cs[i,1] / 10.)**fp[9]
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + loggamma2 - fp[6] * (fb_cs[i,1] / 10.)**fp[7] * np.log10(fb_cs[i,0] / fp[5]) - fp[8] * (fb_cs[i,1] / 10.)**fp[9] * np.log10(1. + fb_cs[i,0] / fp[5]) + fp[14]*(1. - fb_cs[i,0])**fp[15] * np.log10(fb_cs[i,1]))
        model_params[i, 3] = 10**(fp[10] + fp[11]*np.log10(fb_cs[i,0]))
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v23(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 3))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        model_params[i, 0] = 10**(fp[0] + fp[1]*np.log10(fb_cs[i,1]) + fp[2] * (fb_cs[i,1] / 10.)**fp[3] * np.log10(fb_cs[i,0] / fp[4]) + fp[5] * (fb_cs[i,1] / 10.)**fp[6] * np.log10(1. + fb_cs[i,0] / fp[4]))
        model_params[i, 1] = 10**(fp[7] + fp[8]*np.log10(fb_cs[i,1]) + fp[9] * (fb_cs[i,1] / 10.)**fp[10] * np.log10(fb_cs[i,0] / fp[11]) + fp[12] * (fb_cs[i,1] / 10.)**fp[13] * np.log10(1. + fb_cs[i,0] / fp[11]))
        model_params[i, 2] = 10**(fp[14] + fp[15]*np.log10(fb_cs[i,0]))
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v22(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 3))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        model_params[i, 0] = 10**(fp[0] + fp[1] * (fb_cs[i,1] / 10.)**fp[2] * np.log10(fb_cs[i,0] / fp[3]) + fp[4] * (fb_cs[i,1] / 10.)**fp[5] * np.log10(1. + fb_cs[i,0] / fp[3]))
        model_params[i, 1] = 10**(fp[6] + fp[7] * (fb_cs[i,1] / 10.)**fp[8] * np.log10(fb_cs[i,0] / fp[9]) + fp[10] * (fb_cs[i,1] / 10.)**fp[11] * np.log10(1. + fb_cs[i,0] / fp[9]))
        model_params[i, 2] = 10**(fp[12] + fp[13]*np.log10(fb_cs[i,0]))
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v21(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        loggamma1 = np.log10(1. / fp[0]) * fp[1] * (fb_cs[i,1] / 10.)**fp[2] + np.log10(1. + 1./fp[0]) * fp[3] * (fb_cs[i,1] / 10.)**fp[4]
        model_params[i, 0] = 10**(0. + loggamma1 - fp[1] * (fb_cs[i,1] / 10.)**fp[2] * np.log10(fb_cs[i,0] / fp[0]) - fp[3] * (fb_cs[i,1] / 10.)**fp[4] * np.log10(1. + fb_cs[i,0] / fp[0]))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        loggamma2 = np.log10(1. / fp[5]) * fp[6] * (fb_cs[i,1] / 10.)**fp[7] + np.log10(1. + 1./fp[5]) * fp[8] * (fb_cs[i,1] / 10.)**fp[9]
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + loggamma2 - fp[6] * (fb_cs[i,1] / 10.)**fp[7] * np.log10(fb_cs[i,0] / fp[5]) - fp[8] * (fb_cs[i,1] / 10.)**fp[9] * np.log10(1. + fb_cs[i,0] / fp[5]))
        model_params[i, 3] = 10**(fp[10] + fp[11]*np.log10(fb_cs[i,0]))
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v20(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2. + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(fb_cs[i, 0])**3.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        loggamma = np.log10(1. / fp[6]) * fp[7] * (fb_cs[i,1] / 10.)**fp[8] + np.log10(1. + 1./fp[6]) * fp[9] * (fb_cs[i,1] / 10.)**fp[10]
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + loggamma - fp[7] * (fb_cs[i,1] / 10.)**fp[8] * np.log10(fb_cs[i,0] / fp[6]) - fp[9] * (fb_cs[i,1] / 10.)**fp[10] * np.log10(1. + fb_cs[i,0] / fp[6]))
        model_params[i, 3] = 10**(fp[11] + fp[12]*np.log10(fb_cs[i,0]))
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v19(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + (fb_cs[i, 1] / 10.)**fp[0] * (fp[1] * np.log10(fb_cs[i, 0]) + fp[2] * np.log10(fb_cs[i, 0])**2. + fp[3]* np.log10(fb_cs[i, 0])**3))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        loggamma = np.log10(1. / fp[4]) * fp[5] * (fb_cs[i,1] / 10.)**fp[6] + np.log10(1. + 1./fp[4]) * fp[7] * (fb_cs[i,1] / 10.)**fp[8]
        model_params[i, 2] = 10**(np.log10(fb_cs[i,1]) + loggamma - fp[5] * (fb_cs[i,1] / 10.)**fp[6] * np.log10(fb_cs[i,0] / fp[4]) - fp[7] * (fb_cs[i,1] / 10.)**fp[8] * np.log10(1. + fb_cs[i,0] / fp[4]))
        model_params[i, 3] = 10**(fp[9] + fp[10]*np.log10(fb_cs[i,0]))
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v18(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + (fb_cs[i, 1] / 10.)**fp[0] * (fp[1] * np.log10(fb_cs[i, 0]) + fp[2] * np.log10(fb_cs[i, 0])**2. + fp[3]* np.log10(fb_cs[i, 0])**3))
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + (fb_cs[i, 1] / 10.)**fp[4] * (fp[5] * np.log10(fb_cs[i, 0]) + fp[6] * np.log10(fb_cs[i, 0])**2. + fp[7]* np.log10(fb_cs[i, 0])**3 + fp[8]*np.log10(fb_cs[i, 0])**4))
        model_params[i, 3] = 10**(fp[9] + fp[10]*np.log10(fb_cs[i,0]))
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v17(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2. + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(fb_cs[i, 0])**3.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + (fb_cs[i, 1] / 10.)**fp[6] * (fp[7] * np.log10(fb_cs[i, 0]) + fp[8] * np.log10(fb_cs[i, 0])**2. + fp[9]* np.log10(fb_cs[i, 0])**3 + fp[10]*np.log10(fb_cs[i, 0])**4))
        model_params[i, 3] = 10**(fp[11] + fp[12]*np.log10(fb_cs[i,0]))
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v16(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2. + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(fb_cs[i, 0])**3.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(
            fb_cs[i, 0]) + fp[8]*((fb_cs[i, 1] / 10.)**fp[9]) * np.log10(fb_cs[i, 0])**2. + fp[10]*(fb_cs[i, 1] / 10.)**fp[11] * np.log10(fb_cs[i, 0])**3 + fp[12]*(fb_cs[i, 1] / 10.)**fp[13] * np.log10(fb_cs[i, 0])**4 + fp[14]*(fb_cs[i, 1] / 10.)**fp[15] * np.log10(fb_cs[i, 0])**5)
        model_params[i, 3] = 10**(fp[16] + fp[17]*np.log10(fb_cs[i,0]))
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v15(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2. + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(fb_cs[i, 0])**3.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(
            fb_cs[i, 0]) + fp[8]*((fb_cs[i, 1] / 10.)**fp[9]) * np.log10(fb_cs[i, 0])**2. + fp[10]*(fb_cs[i, 1] / 10.)**fp[11] * np.log10(fb_cs[i, 0])**3 + fp[12]*(fb_cs[i, 1] / 10.)**fp[13] * np.log10(fb_cs[i, 0])**4)
        model_params[i, 3] = 10**(fp[14] + fp[15]*np.log10(fb_cs[i,0]) + fp[16]*np.log10(fb_cs[i,0])**2.)
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v14(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2. + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(fb_cs[i, 0])**3.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(
            fb_cs[i, 0]) + fp[8]*((fb_cs[i, 1] / 10.)**fp[9]) * np.log10(fb_cs[i, 0])**2. + fp[10]*(fb_cs[i, 1] / 10.)**fp[11] * np.log10(fb_cs[i, 0])**3 + fp[12]*(fb_cs[i, 1] / 10.)**fp[13] * np.log10(fb_cs[i, 0])**4)
        model_params[i, 3] = 10**(fp[14] + fp[15]*np.log10(fb_cs[i,0]))
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v13(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(
            fb_cs[i, 0]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(fb_cs[i, 0])**2. + fp[8]*(fb_cs[i, 1] / 10.)**fp[9] * np.log10(fb_cs[i, 0])**3 + fp[10]*(fb_cs[i, 1] / 10.)**fp[11] * np.log10(fb_cs[i, 0])**4)
        model_params[i, 3] = 10**(fp[12] + fp[13]*np.log10(fb_cs[i,0]))
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v12(fb_cs, fp):
#this one goes one more order in ft... let's see if third order in both is sufficient
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2. + fp[4]*(fb_cs[i, 1] / 10.)**fp[5] * np.log10(fb_cs[i, 0])**3)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(
            fb_cs[i, 0]) + fp[8]*((fb_cs[i, 1] / 10.)**fp[9]) * np.log10(fb_cs[i, 0])**2. + fp[10]*(fb_cs[i, 1] / 10.)**fp[11] * np.log10(fb_cs[i, 0])**3)
        model_params[i, 3] = 10**(fp[12] + fp[13]*np.log10(fb_cs[i,0]))
    return model_params
    
def paramet_hayashi_deluxe_v11(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(
            fb_cs[i, 0]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(fb_cs[i, 0])**2. + fp[8]*(fb_cs[i, 1] / 10.)**fp[9] * np.log10(fb_cs[i, 0])**3)
        model_params[i, 3] = 10**(fp[10] + fp[11]*np.log10(fb_cs[i,0]) + fp[12]*np.log10(fb_cs[i,0])**2.)
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v11(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(
            fb_cs[i, 0]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(fb_cs[i, 0])**2. + fp[8]*(fb_cs[i, 1] / 10.)**fp[9] * np.log10(fb_cs[i, 0])**3)
        model_params[i, 3] = 10**(fp[10] + fp[11]*np.log10(fb_cs[i,0]) + fp[12]*np.log10(fb_cs[i,0])**2.)
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v10(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(
            fb_cs[i, 0]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(fb_cs[i, 0])**2. + fp[8]*(fb_cs[i, 1] / 10.)**fp[9] * np.log10(fb_cs[i, 0])**3)
        model_params[i, 3] = 10**(fp[10] + fp[11]*(fb_cs[i,1] / 10.)**fp[12] * np.log10(fb_cs[i,0]))
    return model_params

@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v9(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(
            fb_cs[i, 0]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(fb_cs[i, 0])**2. + fp[8]*(fb_cs[i, 1] / 10.)**fp[9] * np.log10(fb_cs[i, 0])**3)
        model_params[i, 3] = 10**(fp[10] + fp[11]*np.log10(fb_cs[i,0]))
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v8(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 3))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # I think that it would make sense to try this with the rho_model_hayashi and hayashi_deluxe
        # does the ability for the transfer function go to unity matter a lot?
        #we'll try this current one with rho_model_hayashi and see if it does better
        model_params[i, 0] = 10**(fp[0] + fp[1]*((fb_cs[i, 1] / 10.)**fp[2]) * np.log10(
            fb_cs[i, 0]) + fp[3]*((fb_cs[i, 1] / 10.)**fp[4]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 1] = 10**(fp[5] + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(
            fb_cs[i, 0]) + fp[8]*((fb_cs[i, 1] / 10.)**fp[9]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 2] = fp[10]
    return model_params
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v7(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(
            fb_cs[i, 0]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 3] = fp[8] + fp[9]*fb_cs[i,0]**fp[10]
    return model_params 
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v6(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(
            fb_cs[i, 0]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 3] = fp[8]
    return model_params 
    
@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v5(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir, but then will want to use hayashi form...
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(
            fb_cs[i, 0]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(fb_cs[i, 0])**2. + fp[8]*(fb_cs[i, 1] / 10.)**fp[9] * np.log10(fb_cs[i, 0])**3)
        model_params[i, 3] = fp[10] + fp[11]*fb_cs[i,0]**fp[12]
    return model_params

@njit(parallel=True)#, fastmath=True)
def paramet_hayashi_deluxe_v4(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(
            fb_cs[i, 0]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(fb_cs[i, 0])**2. + fp[8]*(fb_cs[i, 1] / 10.)**fp[9] * np.log10(fb_cs[i, 0])**3)
        model_params[i, 3] = fp[10]
    return model_params

@njit(parallel=True)#, fastmath=True)
def paramet_hayashi_deluxe_v4(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(
            fb_cs[i, 0]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(fb_cs[i, 0])**2. + fp[8]*(fb_cs[i, 1] / 10.)**fp[9] * np.log10(fb_cs[i, 0])**3)
        model_params[i, 3] = fp[10]
    return model_params


@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v3(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # less computationally efficient, but still works...
        model_params[i, 0] = 10**(0.08574221*(fb_cs[i, 1] / 10.)**-0.35442253 * np.log10(fb_cs[i, 0]
                                                                                         ) + -0.09686971*(fb_cs[i, 1] / 10.)**0.41424968 * np.log10(fb_cs[i, 0])**2)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 3] = fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(
            fb_cs[i, 0])
    return model_params  # lastly, try it logged and see how that goes...


@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v2(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # less computationally efficient, but still works...
        model_params[i, 0] = 10**(0.08574221*(fb_cs[i, 1] / 10.)**-0.35442253 * np.log10(fb_cs[i, 0]
                                                                                         ) + -0.09686971*(fb_cs[i, 1] / 10.)**0.41424968 * np.log10(fb_cs[i, 0])**2)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 3] = fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(
            fb_cs[i, 0]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(fb_cs[i, 0])**2.
    return model_params


@njit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # less computationally efficient, but still works...
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(
            fb_cs[i, 0]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 3] = fp[8]*((fb_cs[i, 1] / 10.)**fp[9]) * np.log10(
            fb_cs[i, 0]) + fp[10]*((fb_cs[i, 1] / 10.)**fp[11]) * np.log10(fb_cs[i, 0])**2.
    return model_params


@njit(parallel=True, fastmath=True)
def paramet_exp_v3_free_ft(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 3))
    for i in prange(0, len(fb_cs)):
        # less computationally efficient, but still works...
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(
            fb_cs[i, 0]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(fb_cs[i, 0])**2.)
    return model_params
    # return np.column_stack((ft_vals, model_params))


@njit(parallel=True, fastmath=True)
def paramet_exp_v3(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 3))
    for i in prange(0, len(fb_cs)):
        # less computationally efficient, but still works...
        model_params[i, 0] = 10**(0.08574221*(fb_cs[i, 1] / 10.)**-0.35442253 * np.log10(fb_cs[i, 0]
                                                                                         ) + -0.09686971*(fb_cs[i, 1] / 10.)**0.41424968 * np.log10(fb_cs[i, 0])**2)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
    return model_params
    # return np.column_stack((ft_vals, model_params))


@njit(parallel=True, fastmath=True)
def hayashi_params(fb_cs, fit_params):
    fb = fb_cs[:, 0]
    # in this case, param_inputs is just the f_b array
    # need as many as there are values of fb, and 3 for ft, rt, delta
    model_params = np.zeros((len(fb), 3))
    for i in prange(0, len(fb)):
        model_params[i, 0] = 10**(fit_params[0] + fit_params[1]*np.log10(
            fb[i]) + fit_params[2]*np.log10(fb[i])**2 + fit_params[3]*np.log10(fb[i])**3)
        model_params[i, 1] = 10**(fit_params[4] + fit_params[5]
                                  * np.log10(fb[i]) + fit_params[6]*np.log10(fb[i])**2)
        model_params[i, 2] = 10**(fit_params[7] + fit_params[8]
                                  * np.log10(fb[i]) + fit_params[9]*np.log10(fb[i])**2)
    return model_params


@njit(parallel=True, fastmath=True)
def hayashi_params_wcs(fb_cs, fit_params):
    # in this case, ip is f_b, c_s arrays
    # need as many as there are values of fb, and 3 for ft, rt, delta
    model_params = np.zeros((len(fb_cs), 3))
    for i in prange(0, len(fb_cs)):
        model_params[i, 0] = 10**(fit_params[0] + fit_params[1]*np.log10(fb_cs[i, 0]) + fit_params[2]*np.log10(
            fb_cs[i, 1]) + fit_params[3]*np.log10(fb_cs[i, 0])**2 + fit_params[4]*np.log10(fb_cs[i, 0])**3)
        model_params[i, 1] = 10**(fit_params[5] + fit_params[6]*np.log10(fb_cs[i, 0]) +
                                  fit_params[7]*np.log10(fb_cs[i, 1]) + fit_params[8]*np.log10(fb_cs[i, 0])**2)
        model_params[i, 2] = 10**(fit_params[9] + fit_params[10]*np.log10(fb_cs[i, 0]) +
                                  fit_params[11]*np.log10(fb_cs[i, 1]) + fit_params[12]*np.log10(fb_cs[i, 0])**2)
    return model_params


@njit(parallel=True, fastmath=True)
def paramet_frank(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 3))
    # passing each time instead of initializing would be faster
    # but then I probably wouldn't be able to parallize as well..
    for i in prange(0, len(fb_cs)):
        model_params[i, 0] = 10**(0.301029995664 + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 1] = 10**(fp[4] + fp[5]*((fb_cs[i, 1] / 10.)**fp[6]) * np.log10(
            fb_cs[i, 0]) + fp[7]*((fb_cs[i, 1] / 10.)**fp[8]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 2] = fp[9]*((fb_cs[i, 1] / 10.)**fp[10]) * np.log10(
            fb_cs[i, 0]) + fp[11]*((fb_cs[i, 1] / 10.)**fp[12]) * np.log10(fb_cs[i, 0])**2
    return model_params


@njit(parallel=True, fastmath=True)
def paramet_ft_fix(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 3))
    for i in prange(0, len(fb_cs)):
        # less computationally efficient, but still works...
        model_params[i, 0] = 10**(0.08574221*(fb_cs[i, 1] / 10.)**-0.35442253 * np.log10(fb_cs[i, 0]
                                                                                         ) + -0.09686971*(fb_cs[i, 1] / 10.)**0.41424968 * np.log10(fb_cs[i, 0])**2)
        model_params[i, 1] = 10**(fp[0] + fp[1]*((fb_cs[i, 1] / 10.)**fp[2]) * np.log10(
            fb_cs[i, 0]) + fp[3]*((fb_cs[i, 1] / 10.)**fp[4]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 2] = fp[5]*((fb_cs[i, 1] / 10.)**fp[6]) * np.log10(
            fb_cs[i, 0]) + fp[7]*((fb_cs[i, 1] / 10.)**fp[8]) * np.log10(fb_cs[i, 0])**2
    return model_params
    # return np.column_stack((ft_vals, model_params))


@njit(parallel=True, fastmath=True)
def paramet_exp(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 2))
    for i in prange(0, len(fb_cs)):
        # less computationally efficient, but still works...
        model_params[i, 0] = 10**(0.08574221*(fb_cs[i, 1] / 10.)**-0.35442253 * np.log10(fb_cs[i, 0]
                                                                                         ) + -0.09686971*(fb_cs[i, 1] / 10.)**0.41424968 * np.log10(fb_cs[i, 0])**2)
        model_params[i, 1] = 10**(fp[0] + fp[1]*((fb_cs[i, 1] / 10.)**fp[2]) * np.log10(
            fb_cs[i, 0]) + fp[3]*((fb_cs[i, 1] / 10.)**fp[4]) * np.log10(fb_cs[i, 0])**2.)
    return model_params
    # return np.column_stack((ft_vals, model_params))


@njit(parallel=True, fastmath=True)
def paramet_ft_fix_nonzero_delt(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 3))
    for i in prange(0, len(fb_cs)):
        # less computationally efficient, but still works...
        model_params[i, 0] = 10**(0.08574221*(fb_cs[i, 1] / 10.)**-0.35442253 * np.log10(fb_cs[i, 0]
                                                                                         ) + -0.09686971*(fb_cs[i, 1] / 10.)**0.41424968 * np.log10(fb_cs[i, 0])**2)
        model_params[i, 1] = 10**(fp[0] + fp[1]*((fb_cs[i, 1] / 10.)**fp[2]) * np.log10(
            fb_cs[i, 0]) + fp[3]*((fb_cs[i, 1] / 10.)**fp[4]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 2] = fp[5] + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(
            fb_cs[i, 0]) + fp[8]*((fb_cs[i, 1] / 10.)**fp[9]) * np.log10(fb_cs[i, 0])**2
    return model_params


@njit(parallel=True, fastmath=True)
def paramet_ft_fix_powerlaw_delt(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 3))
    for i in prange(0, len(fb_cs)):
        # less computationally efficient, but still works...
        model_params[i, 0] = 10**(0.08574221*(fb_cs[i, 1] / 10.)**-0.35442253 * np.log10(fb_cs[i, 0]
                                                                                         ) + -0.09686971*(fb_cs[i, 1] / 10.)**0.41424968 * np.log10(fb_cs[i, 0])**2)
        model_params[i, 1] = 10**(fp[0] + fp[1]*((fb_cs[i, 1] / 10.)**fp[2]) * np.log10(
            fb_cs[i, 0]) + fp[3]*((fb_cs[i, 1] / 10.)**fp[4]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 2] = 10**(fp[5] + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(
            fb_cs[i, 0]) + fp[8]*((fb_cs[i, 1] / 10.)**fp[9]) * np.log10(fb_cs[i, 0])**2)
    return model_params


def rho_fit(r, fb, cs, fit_params, model_param_func, rho_model):
    model_params = model_param_func(np.column_stack((fb, cs)), fit_params)
    if(len(fb) == 1):
        return rho_model(r, model_params[0])  # this is because len(fb) == 1
    else:
        return rho_model(r, model_params)


@njit(parallel=True, fastmath=True)
def cost(fit_parms, model_param_func, rho_model, pdf_weighted=True, error_weighted=True):

    if(pdf_weighted and error_weighted):
        #wt = weight_o_errsq
        pass
    elif(pdf_weighted and not error_weighted):
        #wt = weights_no_err
        pass
    elif(not pdf_weighted and error_weighted):
        #wt = o_errsq
        pass
    elif(not pdf_weighted and not error_weighted):
        wt = unity

    model_params = model_param_func(
        fb_cs, fit_parms)  # use cs doesn't matter now

    # remember that all of these calculations are done in terms of r/r_s,0

    totl = 0
    for i in prange(0, Nsnaps):
        for j in prange(0, n_prof_pts):
            if(o_err[i, j] == 0): #if this point has infinite error, skip it
                continue
            else:
                totl += (wt[i, j] * (np.log10(dat_for_fit[i, 5+j])
                                     - np.log10(rho_model(r_by_rs[i, j], model_params[i])))**2)
            #if(np.isnan(totl)):
            #    print(i,j,totl,np.log10(dat_for_fit[i, 5+j]), rho_model(r_by_rs[i, j], model_params[i]), model_params[i])

    if(error_weighted):
        return totl / 2.  # chi^2 sum([log(rho1) - log(rho2)]**2 / [2 * delta(log(rho1))**2]
    else:
        return totl  # RMSE


def cost_print(fit_parms, model_param_func, rho_model, pdf_weighted=True, error_weighted=True):
    c = cost(fit_parms, model_param_func, rho_model,
             pdf_weighted, error_weighted)
    print(c)
    return c


@njit(parallel=True, fastmath=True)
def lnlik(fit_parms, model_param_func, rho_model, pdf_weighted=True, error_weighted=True):
    for i in range(0, len(fit_parms)):
        if(np.abs(fit_parms[i]) > PRIOR_MAX):
            return -np.inf
    return -1. * cost(fit_parms, model_param_func, rho_model, pdf_weighted, error_weighted)

# delta chi^2 is probably more useful than raw rmse for this metric
@njit(parallel=True, fastmath=True)
def cost_dist(fit_parms, model_param_func, rho_model, chi=True):
    if(chi):
        mt = o_err
    else:
        mt = np.ones(o_err.shape)
    model_params = model_param_func(fb_cs, fit_parms)
    resids = np.zeros((Nsnaps, n_prof_pts))
    for i in prange(0, Nsnaps):
        for j in prange(0, n_prof_pts):
            if(o_err[i, j] == 0):
                continue  # if there is infinite error, then the error-weighted residual is zero
            else:
                resids[i, j] = mt[i, j] * (np.log10(dat_for_fit[i, 5+j]) -
                                           np.log10(rho_model(r_by_rs[i, j], model_params[i])))
    return resids


# for each iteration, we calculate ft, rt, delt at the top of cost, then calculate the cost function
x0 = [-0.007, 0.35, 0.39, 0.23, 1.02, 1.38, 0.37, np.log10(3.), 0., 0.]


# In[43]:


fp_v52_avgnfw_no1Rv_subset = np.array([ 0.35800201,  0.03419161,  0.17848425,  1.3807795 ,  0.43329931,
        0.25070339, -0.19081954,  0.00411616, -1.17201315,  0.11405262,
        0.4465227 , -0.04334982, -0.14059579, -0.11220273,  0.53220652])
print(cost(fp_v52_avgnfw_no1Rv_subset, paramet_hayashi_deluxe_v52, hayashi_deluxe, False, False))

fp_v52_avgnfw_no1Rv_all = np.array([ 3.37821658e-01, -2.21730464e-04,  1.56793984e-01,  1.33726984e+00,
        4.47757739e-01,  2.71551083e-01, -1.98632609e-01,  1.05905814e-02,
       -1.11879075e+00,  9.26587706e-02,  4.43963825e-01, -3.46205146e-02,
       -3.37271922e-01, -9.91000445e-02,  4.14500861e-01])
print(cost(fp_v52_avgnfw_no1Rv_all, paramet_hayashi_deluxe_v52, hayashi_deluxe, False, False))
# should print 160235.76801649653 with all used 1Rv except 10% peris

fp_v52_avgnfw_no1Rv_all_peris = np.array([ 3.26447719e-01, -4.40317402e-04,  1.29767697e-01,  1.17626744e+00,
        4.53968722e-01,  2.13291776e-01, -2.09660878e-01,  5.55304999e-03,
       -1.01125739e+00,  1.25791836e-01,  5.38958395e-01, -8.30901454e-03,
       -7.19380523e-01, -1.18309438e-01,  1.44564797e-10])
print(cost(fp_v52_avgnfw_no1Rv_all_peris, paramet_hayashi_deluxe_v52, hayashi_deluxe, False, False))


# In[44]:


### FIGURE FOR PAPER ###

fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(6,11), sharex=True, gridspec_kw={'hspace':0.025})#(7.5, 5.77))
for i in range(0,3):
    ax[i].yaxis.set_ticks_position('both')
    ax[i].xaxis.set_ticks_position('both')
    ax[i].tick_params(axis='both', which='minor', colors='black', width=1.0, length=2.0)
    ax[i].xaxis.set_minor_locator(MultipleLocator(0.2))
    #if(i==2):
    #    ax[i].semilogx()
    #else:
    #    ax[i].loglog()
ax[0].yaxis.set_minor_locator(MultipleLocator(0.05))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.1))
ax[2].yaxis.set_minor_locator(MultipleLocator(0.1))
            
pfunc = paramet_hayashi_deluxe_v52#paramet_plexp_v2#paramet_hayashi_deluxe_v43
parms = fp_v52_avgnfw_no1Rv_all#fp_plexp_v2_avgnfw_no1Rv_all#fp_v43_avgnfw_no1Rv_all

# this shows why we need to use the priors that I've set...
# so we will try v48 a bit more but with the priors to see if that looks better...

stacked_cols = sns.cubehelix_palette(len(cs_vals))
for i,cs in enumerate(cs_vals):
    fbvals = np.logspace(-3,0.,50)
    fbcs = np.column_stack((fbvals,np.repeat(cs,len(fbvals))))
    pvs = pfunc(fbcs, parms)
    ax[0].plot(np.log10(fbvals), np.log10(pvs[:,0]), label=cs, color=stacked_cols[i])
l = ax[0].legend(title=r'$c_\mathrm{s}$', ncol=2, fontsize=12,frameon=False)
ax[0].get_legend().get_title().set_fontsize('18')
l.set_zorder(20)
#ax[0].set_xlabel(r'$f_\mathrm{b}$')
ax[0].set_ylabel(r'$\log\Big[f_\mathrm{te}\Big]$')
ax[0].set_ylim(np.log10(0.09),np.log10(1.1))

for i,cs in enumerate(cs_vals):
    fbvals = np.logspace(-3,0.,50)
    fbcs = np.column_stack((fbvals,np.repeat(cs,len(fbvals))))
    pvs = pfunc(fbcs, parms)
    ax[1].plot(np.log10(fbvals), np.log10(pvs[:,2]), label=cs, color=stacked_cols[i])
#ax[1].set_xlabel(r'$f_\mathrm{b}$')
ax[1].set_ylabel(r'$\log\Big[\tilde{r}_\mathrm{te}\Big]$')

#ax[1,0].semilogx()
for i,cs in enumerate(cs_vals):
    fbvals = np.logspace(-3,0.,50)
    fbcs = np.column_stack((fbvals,np.repeat(cs,len(fbvals))))
    pvs = pfunc(fbcs, parms)
    ax[2].plot(np.log10(fbvals), pvs[:,3], label=cs, color=stacked_cols[i])
#plt.legend()
ax[2].set_xlabel(r'$\log\Big[f_\mathrm{b}\Big]$')
ax[2].set_ylabel(r'$\delta$')

#for i,cs in enumerate(cs_vals):
#    fbvals = np.logspace(-3,0.,50)
#    fbcs = np.column_stack((fbvals,np.repeat(cs,len(fbvals))))
#    pvs = pfunc(fbcs, parms)
#    ax[1,1].plot(fbvals, pvs[:,4], label=cs, color=stacked_cols[i])
#plt.legend()
#ax[1,1].set_xlabel(r'$f_\mathrm{b}$')
#ax[1,1].set_ylabel(r'$\delta$')
#plt.savefig(fig_dir/'model_param_forms.pdf', bbox_inches='tight')


# ## DASH transfer functions compared to our model, Hayashi and Penarrubia
# 
# ## DASH transfer functions stacked by orbital parameters/concentrations

# In[45]:


#hayashi et al 2003 overplot
def rte(fb):
    return 10**(1.02 + 1.38*np.log10(fb) + 0.37*(np.log10(fb))**2)

def ft(fb):
    return 10**(-0.007 + 0.35*np.log10(fb) + 0.39*(np.log10(fb))**2 + 0.23*(np.log10(fb))**3)

def strip_mod_hayashi(r,fb):
    return ft(fb) / (1. + (r/rte(fb))**3)

#now i just need the original NFW profiles (relaxed and not) for the different simulations cs values...
#find one that has all the cs values and then grab the relevant profiles?


# In[46]:


#penarrubia 2010 method
#haven't yet figured out how to fix this for r_vir,s=1
#I did the calculation originally assuming that r_s,0=1 and haven't been able to convert/fix it yet...

def gen_form(r,rho_0, r_s, alpha, beta, gamma):
    top = rho_0
    bottom = (r/r_s)**gamma * (1+(r/r_s)**alpha)**((beta-gamma)/alpha)
    return top / bottom

#ratio between new vmax/rmax and original vmax/rmax
#x is bound fraction
def g_v_r(x,mu,eta):
    return 2**mu * x**eta / (1+x)**mu

def VmaxNFW(c):
    Vvir = 1. / np.sqrt(c) #model units
    return 0.465 * Vvir * np.sqrt(c / NFWf(c))

def rmaxNFW(c):
    rs = 1.0 #/ c #always in model units
    return 2.163*rs
    
def rsp_solve_nfw(rsp, rmaxp):
    return ((rmaxp/rsp)**2 / (1 + rmaxp/rsp)**2) - NFWf(rmaxp/rsp)

def rsp_solve_b5(rsp, rmaxp):
    return rmaxp**2 + 4*rmaxp*rsp - 3*rsp**2
    #positive root is #

def pen_new_prof(r, fb, cs):
    if(fb <= 0.9):
        return pen_new_prof_b5(r, fb, cs)
    else:
        alpha = 1.0
        beta = 3.0
        gamma = 1.0
        vmp = g_v_r(fb, mu=0.40, eta=0.30)*VmaxNFW(cs)
        rmp = g_v_r(fb, mu=-0.30, eta=0.40)*rmaxNFW(cs)
        rsp = ridder(rsp_solve_nfw, 0.001, 30., args=(rmp))
        rho0p = vmp**2 * rmp / (4 * np.pi * rsp**3 * NFWf(rmp/rsp))
        return(gen_form(r,rho0p, rsp, alpha, beta, gamma))
    
def pen_new_prof_b5(r, fb, cs):
    alpha = 1.0
    beta = 5.0
    gamma = 1.0
    vmp = g_v_r(fb, mu=0.40, eta=0.30)*VmaxNFW(cs) #t=0 was still NFW
    rmp = g_v_r(fb, mu=-0.30, eta=0.40)*rmaxNFW(cs) #t=0 was still NFW
    rsp = ridder(rsp_solve_b5, 0.0001, 30., args=(rmp))
    rho0p = 3*(rmp+rsp)**3 * vmp**2 / (2*np.pi * rmp * (rmp+3*rsp) * rsp**3)
    return(gen_form(r,rho0p, rsp, alpha, beta, gamma))

def rho0(rs,c):
    return 1.0 / (4*np.pi * rs**3 * NFWf(c))

#NFW gamma=1, beta=3, alpha=1
#for vmax, mu=0.40, eta=0.30
#for rmasx mu=-0.30, eta=0.40


# In[47]:


# generate a test, see if g_v_r(f_b) actually gives the same as when we compute the Vmax/Rmax

def penarrubiaMass(r, fb, cs):
    return 4*np.pi*quad(lambda x: x**2 * pen_new_prof(x*cs, fb, cs), 0, r)[0]                                                     

def pen_vm_rm(fb, cs, normed=True):
    #rmax = root(lambda x: 4*np.pi*x**3 * strip_mod_hayashi(x*cs, fb)
    #            * NFWrho(x, cs) - hayashiMass(x, fb, cs), g_v_r(fb, -0.3,0.4)*2.163/cs).x[0]
    rmax = root_scalar(lambda x: 4*np.pi*x**3 * pen_new_prof(x*cs, fb, cs) - penarrubiaMass(x, fb, cs), bracket=(10**-3, 1.2)).root
    vmax = np.sqrt(G*penarrubiaMass(rmax, fb, cs)*cs**3 / rmax)
    # need these as ratios...
    if(normed):
        # root(lambda x: x**3 * NFWrho(x, cs) - NFWmass(x, cs), 2.163/cs).x[0]
        rmax0 = 2.163/cs
        # np.sqrt(G*NFWmass(rmax0, cs) / rmax0) #NFW for IC
        vmax0 = 0.465*np.sqrt(cs/NFWf(cs))
        return rmax/rmax0, vmax/vmax0
    else:
        return rmax, vmax
    
fbv = 0.5
csn = 5
print(pen_vm_rm(fbv, cs_vals[csn]))
print(g_v_r(fbv, -0.3, 0.4)) #rmax
print(g_v_r(fbv, 0.4,0.3)) #vmax
# ah, something is probably wrong with my normalization, but it doesn't matter since we always divide it out
# in the density profile plots
# there is a cs**3 dependence...
# that fixed it! Penarrubia model is self-consistent


# In[48]:


def generate_stacks(bins=5, logfb_min=-2, logfb_max=0., cs_num=5, fmt='mean'):
    if(fmt=='mean'):
        stacked_profiles = np.zeros((bins,n_profile_pts))
        stacked_stds = np.zeros((bins,n_profile_pts))
        mean_cs = np.zeros(bins)
        fb_bin_edges = np.logspace(logfb_min,logfb_max,bins+1)
        fb_bin_centers = 10**((np.log10(fb_bin_edges[1:]) + np.log10(fb_bin_edges[:-1]))*0.5)
        for i in range(0,bins):
            sub_mat = dat_matrix[np.logical_and(dat_matrix[:,4] >= fb_bin_edges[i], dat_matrix[:,4] < fb_bin_edges[i+1])]
            sub_mat = sub_mat[sub_mat[:,1] == cs_vals[cs_num]]
            mean_cs[i] = np.mean(sub_mat[:,1])
            stacked_profiles[i,:] = np.mean(sub_mat[:,5:], axis=0)
            stacked_stds[i,:] = np.std(sub_mat[:,5:], axis=0)
    elif(fmt=='median'):
        stacked_profiles = np.zeros((bins,n_profile_pts))
        stacked_stds = np.zeros((bins,n_profile_pts,2))
        mean_cs = np.zeros(bins)
        fb_bin_edges = np.logspace(logfb_min,logfb_max,bins+1)
        fb_bin_centers = 10**((np.log10(fb_bin_edges[1:]) + np.log10(fb_bin_edges[:-1]))*0.5)
        for i in range(0,bins):
            sub_mat = dat_matrix[np.logical_and(dat_matrix[:,4] >= fb_bin_edges[i], dat_matrix[:,4] < fb_bin_edges[i+1])]
            sub_mat = sub_mat[sub_mat[:,1] == cs_vals[cs_num]] #only taking one cs value
            mean_cs[i] = np.median(sub_mat[:,1])
            stacked_profiles[i,:] = np.median(sub_mat[:,5:], axis=0)
            stacked_stds[i,:] = np.column_stack((stacked_profiles[i,:] - np.quantile(sub_mat[:,5:],0.16, axis=0), np.quantile(sub_mat[:,5:],0.84,axis=0) - stacked_profiles[i,:]))
    return stacked_profiles, stacked_stds, mean_cs, fb_bin_centers, fb_bin_edges

mean_or_median = 'median'
bins=5
csn=5
stacked_profiles, stacked_stds, mean_cs, fb_bin_centers, fb_bin_edges = generate_stacks(bins=bins, logfb_min=-2, logfb_max=0., cs_num=csn, fmt=mean_or_median)


# In[49]:


### FIGURE FOR THE PAPER ###

# this one only works for median and returns stds as differences in log space

def generate_stacks(bins=5, logfb_min=-2, logfb_max=0., cs_num=5, fmt='mean'):
    if(fmt=='median'):
        stacked_profiles = np.zeros((bins,n_profile_pts))
        stacked_stds = np.zeros((bins,n_profile_pts,2))
        mean_cs = np.zeros(bins)
        fb_bin_edges = np.logspace(logfb_min,logfb_max,bins+1)
        fb_bin_centers = 10**((np.log10(fb_bin_edges[1:]) + np.log10(fb_bin_edges[:-1]))*0.5)
        for i in range(0,bins):
            sub_mat = dat_matrix[np.logical_and(dat_matrix[:,4] >= fb_bin_edges[i], dat_matrix[:,4] < fb_bin_edges[i+1])]
            sub_mat = sub_mat[sub_mat[:,1] == cs_vals[cs_num]] #only taking one cs value
            mean_cs[i] = np.median(sub_mat[:,1])
            stacked_profiles[i,:] = np.median(sub_mat[:,5:], axis=0)
            stacked_stds[i,:] = np.column_stack((np.log10(stacked_profiles[i,:]) - np.log10(np.quantile(sub_mat[:,5:],0.16, axis=0)), np.log10(np.quantile(sub_mat[:,5:],0.84,axis=0)) - np.log10(stacked_profiles[i,:])))
    return stacked_profiles, stacked_stds, mean_cs, fb_bin_centers, fb_bin_edges

mean_or_median = 'median'
bins=5
csn=5
stacked_profiles, stacked_stds, mean_cs, fb_bin_centers, fb_bin_edges = generate_stacks(bins=bins, logfb_min=-2, logfb_max=0., cs_num=csn, fmt=mean_or_median)

##### FIGURE FOR THE PAPER #####
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(13,5))
ax[0].yaxis.set_ticks_position('both')
ax[0].xaxis.set_ticks_position('both')
ax[0].tick_params(axis='both', which='minor', colors='black', width=1.0, length=2.0)
#ax[0].loglog()
ax[1].yaxis.set_ticks_position('both')
ax[1].xaxis.set_ticks_position('both')
ax[1].tick_params(axis='both', which='minor', colors='black', width=1.0, length=2.0)
#ax[1].loglog()
stacked_cols = sns.cubehelix_palette(bins)

ax[0].xaxis.set_minor_locator(MultipleLocator(0.1))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.1))
ax[1].xaxis.set_minor_locator(MultipleLocator(0.1))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.1))

mean_or_median = 'median'
bins=5
csn=9
stacked_profiles, stacked_stds, mean_cs, fb_bin_centers, fb_bin_edges = generate_stacks(bins=bins, logfb_min=-1.2, logfb_max=0., cs_num=csn, fmt=mean_or_median)
print(mean_cs)

for i in range(0,bins):
    #plot data
    #NOTE THAT THE ERRORS ARE ALREADY LOGGED FOR THIS FIGURE
    (ln, caps, _) = ax[0].errorbar(np.log10(radii* 10**(0.05*i/bins)), np.log10(stacked_profiles[i,:]), stacked_stds[i,:] if mean_or_median=='mean' else stacked_stds[i,:].T, color=stacked_cols[i],
                 label='$[%.3f, %.3f]$' % (fb_bin_edges[i], fb_bin_edges[i+1]), zorder=bins-i, 
                 marker='o', mfc='white', mew=1.5, linestyle='None', capsize=2)
    for cap in caps:
        cap.set_color(stacked_cols[i])
        cap.set_markeredgewidth(1)

    mod_rhos = np.zeros(len(radii))
    mp = paramet_hayashi_deluxe_v52(np.column_stack((fb_bin_centers[i],mean_cs[i])),fp_v52_avgnfw_no1Rv_all)

    for j in range(0, len(radii)):
        mod_rhos[j] = hayashi_deluxe(radii[j]*mean_cs[i], mp[0])

    #generate penarrubia model
    pen_rhos = np.zeros(len(radii))
    for j in range(0,len(radii)):
        pen_rhos[j] = pen_new_prof(radii[j]*mean_cs[i], fb_bin_centers[i], mean_cs[i]) / gen_form(radii[j]*mean_cs[i],rho0(1.,mean_cs[i]), 1., 1, 3, 1)
    
    #plot hayashi and penarrubia
    if(i==bins-1):
        ax[0].plot(np.log10(radii), np.log10(strip_mod_hayashi(radii*mean_cs[i],fb_bin_centers[i])), 
                 color=stacked_cols[i], linestyle='dotted', label='Hayashi+2003')
        ax[0].plot(np.log10(radii), np.log10(pen_rhos), linestyle='dashed', color=stacked_cols[i],
                 label='Pe{\~ n}arrubia+2010')
        ax[0].plot(np.log10(radii), np.log10(mod_rhos), linestyle='solid', color=stacked_cols[i],
                  label='This work')
    else:
        ax[0].plot(np.log10(radii), np.log10(strip_mod_hayashi(radii*mean_cs[i],fb_bin_centers[i])), color=stacked_cols[i], linestyle='dotted')
        ax[0].plot(np.log10(radii), np.log10(pen_rhos), linestyle='dashed', color=stacked_cols[i])
        ax[0].plot(np.log10(radii), np.log10(mod_rhos), linestyle='solid',color=stacked_cols[i])
        
handles, labels = ax[0].get_legend_handles_labels()
order = [3,4,5,6,7,0,1,2]

ax[0].axvline(np.log10(1. / mean_cs[i]), zorder=-32, color = 'k', ymin=0.7)
ax[0].set_xlabel(r'$\log\Big[r/r_\mathrm{vir,s}\Big]$');
ax[0].set_xlim(np.log10(0.8* 10**-2),np.log10(1.2))
ax[0].set_ylim(np.log10(10**-3),np.log10(1.25*10**0))
ax[0].set_ylabel(r'$\log\Big[H(r | f_\mathrm{b}) = \rho(r | f_\mathrm{b}) / \rho(r,\mathrm{infall})\Big]$')
l = ax[0].legend([handles[i] for i in order], [labels[i] for i in order], title=r'$c_\mathrm{s}=%.0f$, $f_\mathrm{b}=$' %cs_vals[csn],ncol=1, fontsize=12, frameon=False)
ax[0].get_legend().get_title().set_fontsize('18')
l.set_zorder(20)

mean_or_median = 'median'
bins=5
csn=5
stacked_profiles, stacked_stds, mean_cs, fb_bin_centers, fb_bin_edges = generate_stacks(bins=bins, logfb_min=-2., logfb_max=0., cs_num=csn, fmt=mean_or_median)
print(mean_cs)
for i in range(0,bins):
    #plot data
    (ln, caps, _) = ax[1].errorbar(np.log10(radii* 10**(0.05*i/bins)), np.log10(stacked_profiles[i,:]), stacked_stds[i,:] if mean_or_median=='mean' else stacked_stds[i,:].T, color=stacked_cols[i],
                 label='$[%.3f, %.3f]$' % (fb_bin_edges[i], fb_bin_edges[i+1]), zorder=bins-i, 
                 marker='o', mfc='white', mew=1.5, linestyle='None', capsize=2)
    for cap in caps:
        cap.set_color(stacked_cols[i])
        cap.set_markeredgewidth(1)

    mod_rhos = np.zeros(len(radii))
    mp = paramet_hayashi_deluxe_v52(np.column_stack((fb_bin_centers[i],mean_cs[i])),fp_v52_avgnfw_no1Rv_all)

    for j in range(0, len(radii)):
        mod_rhos[j] = hayashi_deluxe(radii[j]*mean_cs[i], mp[0])

    #generate penarrubia model
    pen_rhos = np.zeros(len(radii))
    for j in range(0,len(radii)):
        pen_rhos[j] = pen_new_prof(radii[j]*mean_cs[i], fb_bin_centers[i], mean_cs[i]) / gen_form(radii[j]*mean_cs[i],rho0(1.,mean_cs[i]), 1., 1, 3, 1)
    
    #plot hayashi and penarrubia
    if(i==bins-1):
        ax[1].plot(np.log10(radii), np.log10(strip_mod_hayashi(radii*mean_cs[i],fb_bin_centers[i])), 
                 color=stacked_cols[i], linestyle='dotted', label='Hayashi+2003')
        ax[1].plot(np.log10(radii), np.log10(pen_rhos), linestyle='dashed', color=stacked_cols[i],
                 label='Pe{\~ n}arrubia+2010')
        ax[1].plot(np.log10(radii), np.log10(mod_rhos), linestyle='solid', color=stacked_cols[i],
                  label='This work')
    else:
        ax[1].plot(np.log10(radii), np.log10(strip_mod_hayashi(radii*mean_cs[i],fb_bin_centers[i])), color=stacked_cols[i], linestyle='dotted')
        ax[1].plot(np.log10(radii), np.log10(pen_rhos), linestyle='dashed', color=stacked_cols[i])
        ax[1].plot(np.log10(radii), np.log10(mod_rhos), linestyle='solid',color=stacked_cols[i])
        
ax[1].axvline(np.log10(1. / mean_cs[i]), zorder=-32, color = 'k', ymin=0.3)
ax[1].set_xlabel(r'$\log\Big[r/r_\mathrm{vir,s}\Big]$');
ax[1].set_xlim(np.log10(0.8* 10**-2),np.log10(1.2))
ax[1].set_ylim(np.log10(10**-3),np.log10(1.25*10**0))

handles, labels = ax[1].get_legend_handles_labels()
order = [3,4,5,6,7,0,1,2]
#ax.legend(handles[::-1], labels[::-1], title='Line', loc='upper left')

#ax[1].set_ylabel(r'$H(r | f_\mathrm{b}) = \rho(r | f_\mathrm{b}) / \rho(r,\mathrm{infall})$')
l = ax[1].legend([handles[i] for i in order], [labels[i] for i in order], title=r'$c_\mathrm{s}=%.0f$, $f_\mathrm{b}=$' %cs_vals[csn],ncol=1, fontsize=12, frameon=False)
ax[1].get_legend().get_title().set_fontsize('18')
l.set_zorder(20)
#ax[1].text(0.012, 0.05, r'$c_\mathrm{s}=%.1f$' % cs_vals[csn], fontsize=18)

#plt.savefig(fig_dir / 'stacked_prof_comparison.pdf', bbox_inches='tight')

#do we want to do this for every cs? or at least a few? could make it multiple panels...


# In[74]:


#let's just look at large f_b to see what the slope evolution looks like
bins = 5
stacked_profiles = np.zeros((bins,n_profile_pts))
stacked_stds = np.zeros((bins,n_profile_pts))
mean_cs = np.zeros(bins)
fb_bin_edges = np.linspace(0.7,1.0,bins+1) #ten bins
fb_bin_centers = (fb_bin_edges[1:] + fb_bin_edges[:-1])*0.5
print(fb_bin_centers)
for i in range(0,bins):
    sub_mat = dat_matrix[np.logical_and(dat_matrix[:,4] >= fb_bin_edges[i], dat_matrix[:,4] < fb_bin_edges[i+1])]
    mean_cs[i] = np.mean(sub_mat[:,1])
    stacked_profiles[i,:] = np.mean(sub_mat[:,5:], axis=0)
    stacked_stds[i,:] = np.std(sub_mat[:,5:], axis=0)

fig, ax = loglogplot(figsize=(10,5.8))
stacked_cols = sns.cubehelix_palette(bins)
for i in range(0,bins):
    #plot data
    plt.errorbar(radii*mean_cs[i], stacked_profiles[i,:], stacked_stds[i,:], color=stacked_cols[i],
                 label='%.3f to %.3f' % (fb_bin_edges[i], fb_bin_edges[i+1]))
    
    #generate penarrubia model
    pen_rhos = np.zeros(len(radii))
    for j in range(0,len(radii)):
        pen_rhos[j] = pen_new_prof(radii[j]*mean_cs[i], fb_bin_centers[i], mean_cs[i]) / gen_form(radii[j]*mean_cs[i],rho0(1.,mean_cs[i]), 1., 1, 3, 1)    

plt.xlabel(r'$r/r_\textrm{s,0}$');
plt.xlim(7*10**-3,31.5)#10**0);
plt.ylim(10**-1,1.2*10**0)
#plt.ylabel(r'$\rho(r)$');
plt.ylabel(r'$\rho(r)/\rho(r,\textrm{isolated},t=36\textrm{Gyr})$')
plt.legend(title=r'$f_b$',ncol=2)


# In[75]:


bins = 11 #the 11 different values of eta
stacked_profiles = np.zeros((bins,n_profile_pts))
stacked_stds = np.zeros((bins,n_profile_pts))
fb=0.4
sub_mat = dat_matrix[np.logical_and(dat_matrix[:,4] >= 0.375, dat_matrix[:,4] < 0.425)]
#loop over the different values of f_b and make sure that it seems like the results are consistent
for i in range(0,bins):
    stacked_profiles[i,:] = np.mean(sub_mat[sub_mat[:,3]==eta_vals[i],5:], axis=0)
    stacked_stds[i,:] = np.std(sub_mat[sub_mat[:,3]==eta_vals[i],5:], axis=0)

fig, ax = loglogplot(figsize=(10,5.8))
stacked_cols = sns.cubehelix_palette(bins)
for i in [1,5,10]:#range(0,bins):
    #plot data
    (_, caps, _) = plt.errorbar(radii, stacked_profiles[i,:], stacked_stds[i,:], color=stacked_cols[i],
                 label='%.1f' % eta_vals[i],capsize=5)
    for cap in caps:
        cap.set_color(stacked_cols[i])
        cap.set_markeredgewidth(2)    

plt.xlabel(r'$r/r_\textrm{vir,s}$');
plt.xlim(10**-2,10**0);
plt.ylim(3*10**-3,1.2*10**0)
#plt.ylabel(r'$\rho(r)$');
plt.ylabel(r'$\rho(r)/\rho(r,t=0)$')
legend = plt.legend(title=r'$f_b=0.4\pm 0.025$, $\eta$ values:',ncol=2)
legend.get_title().set_fontsize('18')
#plt.savefig(fig_dir/'eta_evolve_fb0.4.pdf',bbox_inches='tight')

# this suggests that for fixed fb, the more circular the orbit, the more of the material that comes from the outside
# of the profile


# In[76]:


bins = 11 #the 11 different values of c_h
stacked_profiles = np.zeros((bins,n_profile_pts))
stacked_stds = np.zeros((bins,n_profile_pts))
fb=0.4
sub_mat = dat_matrix[np.logical_and(dat_matrix[:,4] >= 0.375, dat_matrix[:,4] < 0.425)]
#loop over the different values of f_b and make sure that it seems like the results are consistent
for i in range(0,bins):
    stacked_profiles[i,:] = np.mean(sub_mat[sub_mat[:,0]==ch_vals[i],5:], axis=0)
    stacked_stds[i,:] = np.std(sub_mat[sub_mat[:,0]==ch_vals[i],5:], axis=0)

fig, ax = loglogplot(figsize=(10,5.8))
stacked_cols = sns.cubehelix_palette(bins)
for i in [1,3,5]:#range(0,bins):
    #plot data
    (_, caps, _) = plt.errorbar(radii, stacked_profiles[i,:], stacked_stds[i,:], color=stacked_cols[i],
                 label='%.1f' % ch_vals[i],capsize=5)
    for cap in caps:
        cap.set_color(stacked_cols[i])
        cap.set_markeredgewidth(2)    

plt.xlabel(r'$r/r_\textrm{vir,s}$');
plt.xlim(10**-2,10**0);
plt.ylim(3*10**-3,1.2*10**0)
#plt.ylabel(r'$\rho(r)$');
plt.ylabel(r'$\rho(r)/\rho(r,t=0)$')
legend = plt.legend(title=r'$f_b=0.4\pm 0.025$, $c_h$ values:',ncol=2)
legend.get_title().set_fontsize('18')
#plt.savefig(fig_dir/'ch_evolve_fb0.4.pdf',bbox_inches='tight')

# this suggests that overall, there is minimal dependence on host halo concentation
# however, it does make a difference
# unsurprisingly, a more concentrated host results in an overall more stripped outer profile


# In[54]:


bins = 11 #the 11 different values of c_s
stacked_profiles = np.zeros((bins,n_profile_pts))
stacked_stds = np.zeros((bins,n_profile_pts))
fb=0.3
sub_mat = dat_matrix[np.logical_and(dat_matrix[:,4] >= 0.275, dat_matrix[:,4] < 0.325)]
#loop over the different values of f_b and make sure that it seems like the results are consistent
for i in range(0,bins):
    stacked_profiles[i,:] = np.mean(sub_mat[sub_mat[:,1]==cs_vals[i],5:], axis=0)
    stacked_stds[i,:] = np.std(sub_mat[sub_mat[:,1]==cs_vals[i],5:], axis=0)
fig, ax = loglogplot(figsize=(10,5.8))
stacked_cols = sns.cubehelix_palette(bins)
for i in [1,5,10]:#range(0,bins):
    #plot data
    (_, caps, _) = plt.errorbar(radii[radii<1.], stacked_profiles[i,:][radii<1.], stacked_stds[i,:][radii<1.], color=stacked_cols[i],
                 label='%.1f' % cs_vals[i],capsize=5)
    print(np.log10(stacked_profiles[i,:][radii<1.]))
    for cap in caps:
        cap.set_color(stacked_cols[i])
        cap.set_markeredgewidth(2)    

plt.xlabel(r'$r/r_\textrm{vir,s}$');
#plt.xlim(10**-2, 1.)
#plt.xlim(7*10**-3,31.5)#10**0);
plt.ylim(3*10**-3,1.2*10**0)
#plt.ylabel(r'$\rho(r)$');
plt.ylabel(r'$\rho(r)/\rho(r,\textrm{isolated})$')
legend = plt.legend(title=r'$f_b=0.3\pm 0.025$, $c_s$ values:',ncol=2)
legend.get_title().set_fontsize('18')
plt.savefig(fig_dir/'cs_evolve_fb0.3.pdf',bbox_inches='tight')#0.542208322455#0.492065425614#0.89970615654

# must substantially, this shows that more concentrated haloes are stripped further in for the same amount of mass lost


# In[80]:


bins = 11 #the 11 different values of x_c
stacked_profiles = np.zeros((bins,n_profile_pts))
stacked_stds = np.zeros((bins,n_profile_pts))
fb=0.4
sub_mat = dat_matrix[np.logical_and(dat_matrix[:,4] >= 0.375, dat_matrix[:,4] < 0.425)]
#loop over the different values of f_b and make sure that it seems like the results are consistent
for i in range(0,bins):
    stacked_profiles[i,:] = np.mean(sub_mat[sub_mat[:,2]==xc_vals[i],5:], axis=0)
    stacked_stds[i,:] = np.std(sub_mat[sub_mat[:,2]==xc_vals[i],5:], axis=0)

fig, ax = loglogplot(figsize=(10,5.8))
stacked_cols = sns.cubehelix_palette(bins)
for i in [1,5,10]:#range(0,bins):
    #plot data
    (_, caps, _) = plt.errorbar(radii, stacked_profiles[i,:], stacked_stds[i,:], color=stacked_cols[i],
                 label='%.1f' % xc_vals[i],capsize=5)
    for cap in caps:
        cap.set_color(stacked_cols[i])
        cap.set_markeredgewidth(2)    

plt.xlabel(r'$r/r_\textrm{vir,s}$');
plt.xlim(10**-2,10**0);
plt.ylim(3*10**-3,1.2*10**0)
#plt.ylabel(r'$\rho(r)$');
plt.ylabel(r'$\rho(r)/\rho(r,t=0)$')
legend = plt.legend(title=r'$f_b=0.4\pm 0.025$, $x_c$ values:',ncol=2)
legend.get_title().set_fontsize('18')
#plt.savefig(fig_dir/'xc_evolve_fb0.4.pdf',bbox_inches='tight')

# xc has very little effect


# ## Transfer function residuals versus orbital parameters/concentrations/models

# In[55]:


# let's generate the (static) comparisons (residuals) from Hayashi and Penarrubia
fp_hayashi = [-0.007, 0.35, 0.39, 0.23, 1.02, 1.38, 0.37, np.log10(3.), 0., 0.]
hay_resids = cost_dist(fp_hayashi, hayashi_params, rho_model_hayashi, chi=False)
#this throws out the places with infinite error


# In[56]:


hay_resids.shape


# ### Calculating Penarrubia's transfer functions

# In[77]:


from joblib import Parallel, delayed
from tqdm import tqdm_notebook

# will need to re-run this

def pen_model(a,b):
    return pen_new_prof(r_by_rs[a, b], fb_cs[a,0], fb_cs[a,1]) /         gen_form(r_by_rs[a, b], rho0(1., fb_cs[a,1]), 1., 1, 3, 1)

pms = Parallel(n_jobs=8)(delayed(pen_model)(a=i, b=j) for i in tqdm_notebook(range(0, Nsnaps)) for j in range(0, n_prof_pts))

# should we include the fb=1.0 in the error distribution? Should tighten up our error bars

# TODO: re-run making our dat_for_fit with the fb=1.0 added back in

#pen_resids = cost_dist_pen(chi=False)

#plot_resids(None, None, None, pen=True, chi=False)
    
#need to calculate this for every point in dash and get the cost...


# In[78]:


pms = np.array(pms).reshape(Nsnaps, n_prof_pts)


# In[79]:


np.save('penarrubia_transfer_functions',pms)


# ### Comparison of our model residuals to Hayashi, Penarrubia

# In[57]:


pms = np.load('penarrubia_transfer_functions.npy')


# In[58]:


pms.shape
#not sure where the different number of total snapshots came from
#should be able to slice out the relevant radii but need to diff 573295 from 579528...


# In[59]:


pen_resids = (np.log10(dat_for_fit[:, 5:]) - np.log10(pms))
pen_resids[o_err == 0] = 0


# In[61]:


######### FIGURE FOR PAPER #########
###### TODO: RERUN PENARRUBIA MODELS FOR CURRENT SET OF SNAPS x RADS #####

#now we've got our penarrubia model, let's look at the distribution
def model_compare_resids(fp, param_func, model_func):
    model_resids = cost_dist(fp, param_func, model_func, chi=False)
    labs = ['This work', 'Pe{\~ n}arrubia+2010', 'Hayashi+2003']
    ls = ['solid', 'dashed', 'dotted']
    cls = ['b', 'k', 'r']
    fig, ax = plot(semilogx=True)
    #cols = sns.cubehelix_palette(3)
    for num,resids in enumerate([model_resids, pen_resids, hay_resids]):
        resids = resids * -1.  #switching to model - data
        dist_means = np.zeros(n_prof_pts)
        dist_low16 = np.zeros(n_prof_pts)
        dist_low2 = np.zeros(n_prof_pts)
        dist_high84 = np.zeros(n_prof_pts)
        dist_high97 = np.zeros(n_prof_pts)
        for i in range(0, n_prof_pts):
            dist_means[i] = np.mean(resids[resids[:,i] != 0,i])
            dist_low16[i] = dist_means[i] - np.percentile(resids[resids[:,i] != 0,i], 16)
            dist_low2[i] =  dist_means[i] - np.percentile(resids[resids[:,i] != 0,i], 2.26)
            dist_high84[i] = np.percentile(resids[resids[:,i] != 0,i], 84) - dist_means[i]
            dist_high97[i] = np.percentile(resids[resids[:,i] != 0,i], 97.74) - dist_means[i]
    
        (_, caps, _) = ax.errorbar(fit_rads* 10**(0.05*num/3), dist_means, yerr=[dist_low16, dist_high84], fmt='.-', linestyle=ls[num], color=cls[num], label = labs[num], zorder=3-num, capsize=2, markersize=10)
        for cap in caps:
            #cap.set_color(cols[num])
            cap.set_markeredgewidth(1)
    ax.set_xlabel(r'$r/r_\mathrm{vir,s}$')
    ax.set_ylabel(r'$\log(H_\mathrm{m}) - \log(H_\mathrm{s})$')
    #plt.ylim(np.min(dist_low2), np.max(dist_high97))
    l = ax.legend(title=r'Model', fontsize=12, frameon=False)
    ax.get_legend().get_title().set_fontsize('18')
    l.set_zorder(20)
    plt.axhline(0.,color='k')
    #plt.savefig(fig_dir/'res_plot_model_compare.eps', bbox_inches='tight')
    return fig,ax

# transfer functions are normalized by bin-avg'd NFW, causing H_s to be larger than if they were normalized by NFW instead
# if we used NFW instead

# Frank may ask us to do this out to Rvir... may as well go ahead and do it


# In[62]:


model_compare_resids(fp_v52_avgnfw_no1Rv_all, paramet_hayashi_deluxe_v52, hayashi_deluxe)


# In[63]:


### FIGURE FOR PAPER ###
# plot all of the figures in a 3x2

def plot_resids_all(fp, param_func, model_func, fb_bins=5, logfb_min=-2, logfb_max=0.):
    fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(13,9), sharex=True, sharey=True, gridspec_kw={'wspace':0.05,'hspace':0.075})
    for i in range(0,2):
        for j in range(0,3):
            ax[i,j].yaxis.set_ticks_position('both')
            ax[i,j].xaxis.set_ticks_position('both')
            ax[i,j].tick_params(axis='both', which='minor', colors='black', width=1.0, length=2.0)
            #ax[i,j].semilogx()
            ax[i,j].set_ylim(-0.77,0.77)
            ax[i,j].set_xlim(np.log10(0.8*1e-2),np.log10(1*1.2))
            ax[i,j].xaxis.set_minor_locator(MultipleLocator(0.2))
            ax[i,j].yaxis.set_minor_locator(MultipleLocator(0.05))
    #start with fb on the left...
    fb_bin_edges = np.logspace(logfb_min,logfb_max,fb_bins+1)
    fb_bin_centers = 10**((np.log10(fb_bin_edges[1:]) + np.log10(fb_bin_edges[:-1]))*0.5)
    full_resids = cost_dist(fp, param_func, model_func, chi=False)
    cols = sns.cubehelix_palette(fb_bins)
    for num in range(0,fb_bins):
        resids = full_resids[np.logical_and(dat_for_fit[:,4] > fb_bin_edges[num], dat_for_fit[:,4]<=fb_bin_edges[num+1])]
        resids = resids * -1. #switching to model - data
        print(len(resids))
        dist_means = np.zeros(n_prof_pts)
        dist_low16 = np.zeros(n_prof_pts)
        dist_low2 = np.zeros(n_prof_pts)
        dist_high84 = np.zeros(n_prof_pts)
        dist_high97 = np.zeros(n_prof_pts)
        for i in range(0, n_prof_pts):
            dist_means[i] = np.median(resids[resids[:,i] != 0,i])
            dist_low16[i] = dist_means[i] - np.percentile(resids[resids[:,i] != 0,i], 16)
            dist_low2[i] =  dist_means[i] - np.percentile(resids[resids[:,i] != 0,i], 2.26)
            dist_high84[i] = np.percentile(resids[resids[:,i] != 0,i], 84) - dist_means[i]
            dist_high97[i] = np.percentile(resids[resids[:,i] != 0,i], 97.74) - dist_means[i]



        (_, caps, _) = ax[0,0].errorbar(np.log10(fit_rads* 10**(0.05*num/fb_bins)), dist_means, yerr=[dist_low16, dist_high84], color=cols[num], label = '$[%.3f, %.3f]$' %(fb_bin_edges[num], fb_bin_edges[num+1]),zorder=fb_bins-num, capsize=2)
        for cap in caps:
            cap.set_color(cols[num])
            cap.set_markeredgewidth(1)
            
            
    handles, labels = ax[0,0].get_legend_handles_labels()
    for lj in range(0,3):
        handles.insert(3,mlines.Line2D([],[],linestyle=''))
        labels.insert(3,'')
    


    #plt.errorbar(fit_rads, dist_means, yerr=[dist_low2, dist_high97], alpha=0.5, label='2.5/97.5')
    #ax[0,0].set_xlabel(r'$r/r_\mathrm{vir,s}$')
    ax[0,0].set_ylabel(r'$\log(H_\mathrm{m}) - \log(H_\mathrm{D})$')
    #plt.ylim(np.min(dist_low2), np.max(dist_high97))
    ax[0,0].axhline(0.,color='k')
    l = ax[0,0].legend(handles, labels, title=r'$f_\mathrm{b}$', fontsize=12, frameon=False, loc=2, ncol=1)
    ax[0,0].get_legend().get_title().set_fontsize('18')
    l.set_zorder(20)
    
    #now we do cs
    cols = sns.cubehelix_palette(len(cs_vals))
    #ax[1,0].semilogx()
    for num in range(0,len(cs_vals)):
        resids = full_resids[dat_for_fit[:,1] == cs_vals[num]] * -1. #switching to model - data
        dist_means = np.zeros(n_prof_pts)
        dist_low16 = np.zeros(n_prof_pts)
        dist_low2 = np.zeros(n_prof_pts)
        dist_high84 = np.zeros(n_prof_pts)
        dist_high97 = np.zeros(n_prof_pts)
        for i in range(0, n_prof_pts):
            dist_means[i] = np.median(resids[resids[:,i] != 0,i])
            dist_low16[i] = dist_means[i] - np.percentile(resids[resids[:,i] != 0,i], 16)
            dist_low2[i] =  dist_means[i] - np.percentile(resids[resids[:,i] != 0,i], 2.26)
            dist_high84[i] = np.percentile(resids[resids[:,i] != 0,i], 84) - dist_means[i]
            dist_high97[i] = np.percentile(resids[resids[:,i] != 0,i], 97.74) - dist_means[i]


        #TODO: FIX THE ZORDER
        (_, caps, _) = ax[1,0].errorbar(np.log10(fit_rads* 10**(0.05*num/len(cs_vals))), dist_means, yerr=[dist_low16, dist_high84], color=cols[num], label = '%.1f' %(cs_vals[num]), capsize=2)
        for cap in caps:
            cap.set_color(cols[num])
            cap.set_markeredgewidth(1)
    #plt.errorbar(fit_rads, dist_means, yerr=[dist_low2, dist_high97], alpha=0.5, label='2.5/97.5')
    ax[1,0].axhline(0.,color='k')
    ax[1,0].set_xlabel(r'$\log\Big[r/r_\mathrm{vir,s}\Big]$')
    ax[1,0].set_ylabel(r'$\log(H_\mathrm{m}) - \log(H_\mathrm{D})$')
    #ax[1].set_ylabel(r'$\log(H_\textrm{s}) - \log(H_\textrm{m})$')
    #plt.ylim(np.min(dist_low2), np.max(dist_high97))
    l = ax[1,0].legend(title=r'$c_\mathrm{s}$', loc=3,ncol=3, fontsize=12, frameon=False)
    ax[1,0].get_legend().get_title().set_fontsize('18')
    l.set_zorder(20)
    
        #now we do c_h
    #now we do eta
    ch1 = 0
    chf = len(eta_vals) - 1
    cols = sns.cubehelix_palette(chf - ch1 + 1)
    #ax[0,1].semilogx()
    for num in range(ch1,chf+1):
        resids = full_resids[dat_for_fit[:,0] == ch_vals[num]] * -1. #switching to model - data
        dist_means = np.zeros(n_prof_pts)
        dist_low16 = np.zeros(n_prof_pts)
        dist_low2 = np.zeros(n_prof_pts)
        dist_high84 = np.zeros(n_prof_pts)
        dist_high97 = np.zeros(n_prof_pts)
        for i in range(0, n_prof_pts):
            dist_means[i] = np.median(resids[resids[:,i] != 0,i])
            dist_low16[i] = dist_means[i] - np.percentile(resids[resids[:,i] != 0,i], 16)
            dist_low2[i] =  dist_means[i] - np.percentile(resids[resids[:,i] != 0,i], 2.26)
            dist_high84[i] = np.percentile(resids[resids[:,i] != 0,i], 84) - dist_means[i]
            dist_high97[i] = np.percentile(resids[resids[:,i] != 0,i], 97.74) - dist_means[i]


        #TODO: FIX THE ZORDER
        (_, caps, _) = ax[0,1].errorbar(np.log10(fit_rads* 10**(0.05*num/len(ch_vals))), dist_means, yerr=[dist_low16, dist_high84], color=cols[num-ch1], label = '%.1f' %(ch_vals[num]), capsize=2)
        for cap in caps:
            cap.set_color(cols[num-ch1])
            cap.set_markeredgewidth(1)
    #plt.errorbar(fit_rads, dist_means, yerr=[dist_low2, dist_high97], alpha=0.5, label='2.5/97.5')
    #ax[0,1].set_xlabel(r'$r/r_\mathrm{vir,s}$')
    #ax[0,1].set_ylabel(r'$\log(H_\mathrm{m}) - \log(H_\mathrm{s})$')
    ax[0,1].axhline(0.,color='k')
    #ax[1].set_ylabel(r'$\log(H_\textrm{s}) - \log(H_\textrm{m})$')
    #plt.ylim(np.min(dist_low2), np.max(dist_high97))
    l = ax[0,1].legend(title=r'$c_\mathrm{h}$', loc=2,ncol=3, fontsize=12, frameon=False)
    ax[0,1].get_legend().get_title().set_fontsize('18')
    l.set_zorder(20)
    
    
    xc1 = 2
    xcf = len(xc_vals) - 1
    cols = sns.cubehelix_palette(xcf - xc1 + 1)
    #ax[1,1].semilogx()
    for num in range(xc1,xcf+1):
        resids = full_resids[dat_for_fit[:,2] == xc_vals[num]] * -1. #switching to model - data
        dist_means = np.zeros(n_prof_pts)
        dist_low16 = np.zeros(n_prof_pts)
        dist_low2 = np.zeros(n_prof_pts)
        dist_high84 = np.zeros(n_prof_pts)
        dist_high97 = np.zeros(n_prof_pts)
        for i in range(0, n_prof_pts):
            dist_means[i] = np.median(resids[resids[:,i] != 0,i])
            dist_low16[i] = dist_means[i] - np.percentile(resids[resids[:,i] != 0,i], 16)
            dist_low2[i] =  dist_means[i] - np.percentile(resids[resids[:,i] != 0,i], 2.26)
            dist_high84[i] = np.percentile(resids[resids[:,i] != 0,i], 84) - dist_means[i]
            dist_high97[i] = np.percentile(resids[resids[:,i] != 0,i], 97.74) - dist_means[i]


        #TODO: FIX THE ZORDER
        (_, caps, _) = ax[1,1].errorbar(np.log10(fit_rads* 10**(0.05*num/len(xc_vals))), dist_means, yerr=[dist_low16, dist_high84], color=cols[num-xc1], label = '%.1f' %(xc_vals[num]), capsize=2)
        for cap in caps:
            cap.set_color(cols[num-xc1])
            cap.set_markeredgewidth(1)
    #plt.errorbar(fit_rads, dist_means, yerr=[dist_low2, dist_high97], alpha=0.5, label='2.5/97.5')
    ax[1,1].set_xlabel(r'$\log\Big[r/r_\mathrm{vir,s}\Big]$')
    #ax[0,1].set_ylabel(r'$\log(H_\mathrm{m}) - \log(H_\mathrm{s})$')
    ax[1,1].axhline(0.,color='k')
    #plt.ylim(np.min(dist_low2), np.max(dist_high97))
    l = ax[1,1].legend(title=r'$x_\mathrm{c}$', loc=3, fontsize=12, frameon=False, ncol=3)
    ax[1,1].get_legend().get_title().set_fontsize('18')
    l.set_zorder(20)
    
    #now we do eta
    et1 = 1
    etf = len(eta_vals) - 1
    cols = sns.cubehelix_palette(etf - et1 + 1)
    #ax[0,2].semilogx()
    for num in range(et1,etf+1):
        resids = full_resids[dat_for_fit[:,3] == eta_vals[num]] * -1. #switching to model - data
        dist_means = np.zeros(n_prof_pts)
        dist_low16 = np.zeros(n_prof_pts)
        dist_low2 = np.zeros(n_prof_pts)
        dist_high84 = np.zeros(n_prof_pts)
        dist_high97 = np.zeros(n_prof_pts)
        for i in range(0, n_prof_pts):
            dist_means[i] = np.median(resids[resids[:,i] != 0,i])
            dist_low16[i] = dist_means[i] - np.percentile(resids[resids[:,i] != 0,i], 16)
            dist_low2[i] =  dist_means[i] - np.percentile(resids[resids[:,i] != 0,i], 2.26)
            dist_high84[i] = np.percentile(resids[resids[:,i] != 0,i], 84) - dist_means[i]
            dist_high97[i] = np.percentile(resids[resids[:,i] != 0,i], 97.74) - dist_means[i]


        #TODO: FIX THE ZORDER
        (_, caps, _) = ax[0,2].errorbar(np.log10(fit_rads* 10**(0.05*num/len(eta_vals))), dist_means, yerr=[dist_low16, dist_high84], color=cols[num-et1], label = '%.1f' %(eta_vals[num]), capsize=2)
        for cap in caps:
            cap.set_color(cols[num-et1])
            cap.set_markeredgewidth(1)
    #plt.errorbar(fit_rads, dist_means, yerr=[dist_low2, dist_high97], alpha=0.5, label='2.5/97.5')
    #ax[1].set_xlabel(r'$r/r_\mathrm{vir,s}$')
    #ax[1,1].set_ylabel(r'$\log(H_\mathrm{m}) - \log(H_\mathrm{s})$')
    ax[0,2].axhline(0.,color='k')
    #ax[1].set_ylabel(r'$\log(H_\textrm{s}) - \log(H_\textrm{m})$')
    #plt.ylim(np.min(dist_low2), np.max(dist_high97))
    l = ax[0,2].legend(title=r'$\eta$', loc=2,ncol=3, fontsize=12, frameon=False)
    ax[0,2].get_legend().get_title().set_fontsize('18')
    l.set_zorder(20)
    
    model_resids = cost_dist(fp, param_func, model_func, chi=False)
    labs = ['This work', 'Pe{\~ n}arrubia+2010', 'Hayashi+2003']
    ls = ['solid', 'dashed', 'dotted']
    cls = ['b', 'k', 'r']
    #cols = sns.cubehelix_palette(3)
    for num,resids in enumerate([model_resids, pen_resids, hay_resids]):
        resids = resids * -1.  #switching to model - data
        dist_means = np.zeros(n_prof_pts)
        dist_low16 = np.zeros(n_prof_pts)
        dist_low2 = np.zeros(n_prof_pts)
        dist_high84 = np.zeros(n_prof_pts)
        dist_high97 = np.zeros(n_prof_pts)
        for i in range(0, n_prof_pts):
            dist_means[i] = np.median(resids[resids[:,i] != 0,i])
            dist_low16[i] = dist_means[i] - np.percentile(resids[resids[:,i] != 0,i], 16)
            dist_low2[i] =  dist_means[i] - np.percentile(resids[resids[:,i] != 0,i], 2.26)
            dist_high84[i] = np.percentile(resids[resids[:,i] != 0,i], 84) - dist_means[i]
            dist_high97[i] = np.percentile(resids[resids[:,i] != 0,i], 97.74) - dist_means[i]
    
        (_, caps, _) = ax[1,2].errorbar(np.log10(fit_rads* 10**(0.05*num/3)), dist_means, yerr=[dist_low16, dist_high84], fmt='.-', linestyle=ls[num], color=cls[num], label = labs[num], zorder=3-num, capsize=2, markersize=10)
        for cap in caps:
            #cap.set_color(cols[num])
            cap.set_markeredgewidth(1)
    ax[1,2].set_xlabel(r'$\log\Big[r/r_\mathrm{vir,s}\Big]$')
    #ax[2,1].set_ylabel(r'$\log(H_\mathrm{m}) - \log(H_\mathrm{s})$')
    #plt.ylim(np.min(dist_low2), np.max(dist_high97))
    l = ax[1,2].legend(title=r'Model', fontsize=12, frameon=False)
    ax[1,2].get_legend().get_title().set_fontsize('18')
    l.set_zorder(20)
    plt.axhline(0.,color='k')
    
    # set all of the y-axes to be the same
    
    
    #plt.savefig(fig_dir/'res_plot_all.pdf', bbox_inches='tight')


# In[64]:


plot_resids_all(fp_v52_avgnfw_no1Rv_all, paramet_hayashi_deluxe_v52, hayashi_deluxe, fb_bins=5, logfb_min=-2.5, logfb_max=0.)


# ## Structural parameter comparisons and fitting function

# In[65]:


max_mat = np.zeros((timesteps*nsims, 7+len(radii)))#[] #preload this as numpy zeros of the proper size
vm_rm_arr = np.zeros((timesteps*nsims, 2))

for i in range(0,11):
    print("i",i)
    for j in range(0,11):
        print("j",j)
        for k in range(0,11):
            for l in range(0,11):
                direct = dash_root / ch_dirs[i] / cs_dirs[j] / xc_dirs[k] / eta_dirs[l]
                pf_fn = direct / 'radprof_m.txt'
                sh_evo_fn = direct / 'subhalo_evo.txt'
                if(Path(pf_fn).is_file()):
                    sim_dat, vm_rm = load_vrmax(direct, j, normed=True)#[0,:][np.newaxis,:]
                    snaps_used = sim_dat.shape[0]
                    row = ((i*11**3)+(j*11**2)+(k*11)+l)*timesteps
                    max_mat[row:row+snaps_used,0] = ch_vals[i]
                    max_mat[row:row+snaps_used,1] = cs_vals[j]
                    max_mat[row:row+snaps_used,2] = xc_vals[k]
                    max_mat[row:row+snaps_used,3] = eta_vals[l]
                    max_mat[row:row+snaps_used,4:] = sim_dat
                    vm_rm_arr[row:row+snaps_used] = vm_rm
                
                    
max_mat = max_mat[~np.all(max_mat == 0, axis=1)]
vm_rm_arr = vm_rm_arr[~np.all(vm_rm_arr == 0, axis=1)]
assert(vm_rm_arr.shape[0] == max_mat.shape[0])
#ch, cs, xc, eta, fb, vmax, rmax (normed to vmax/rmax at t=0)

# thus, we are normalizing by t=0 snapshot for our data, and for our model (which is trained on data normalized to avg nfw)
# whereas we are normalizing hayashi results by NFW and penarrubia results come normalized already

# can mention that the IC/t=0 results of dash differ from NFW by less than 1% for rmax and are virtually identical for vmax


# In[66]:


@njit
def exp_decay_v3(r, mp):
    #f_t, c_s, and r_t
    return mp[0] * np.exp(-1.* r * ((mp[1] - mp[2])/(mp[1]*mp[2])))

@njit(parallel = True, fastmath = True)
def paramet_exp_v3_free_ft(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 3))
    for i in prange(0, len(fb_cs)):
        #less computationally efficient, but still works...
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 1]=fb_cs[i,1] #just c_s
        model_params[i, 2]=10**(np.log10(fb_cs[i,1]) + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(
            fb_cs[i, 0]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(fb_cs[i, 0])**2.)
    return model_params
    #return np.column_stack((ft_vals, model_params))
    
@jit
def hayashi_deluxe(r, mp):
    # f_t, r_vir (i.e., c_s), r_t, delta
    if(np.abs(mp[1]-mp[2]) < 1e-8):
        #print("this happened for r=%.3e"%r)
        return mp[0]
    else:
        return mp[0] / (1. + (r * ((mp[1] - mp[2])/(mp[1]*mp[2])))**mp[3])
    
@jit
def hayashi_deluxe_deluxe(r, mp):
    # f_t, r_vir (i.e., c_s), r_t, delta
    if(np.abs(mp[1]-mp[2]) < 1e-8):
        return mp[0]
    elif(mp[1] < mp[2]): #rt is larger than rvir, which shouldn't be the case...
        return 0. #this will throw an inf
    else:
        return mp[0] / (1. + (r * ((mp[1] - mp[2])/(mp[1]*mp[2])))**mp[3])**mp[4]    
    
@jit
def powerlaw_exp(r, mp):
    if(np.abs(mp[1]-mp[2]) < 1e-8):
        #print("this happened for r=%.3e"%r)
        return mp[0]
    elif(mp[1] < mp[2]): #rt is larger than rvir, which shouldn't be the case...
        return 0. #this will throw an inf
    else:
        return (mp[0] / (1. + (r * ((mp[1] - mp[2])/(mp[1]*mp[2])))**mp[3])) * np.exp(-1. * r * ((mp[1] - mp[4])/(mp[1]*mp[4])))


@jit(parallel=True, fastmath=True)
def paramet_hayashi_deluxe_v5(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(
            fb_cs[i, 0]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(fb_cs[i, 0])**2. + fp[8]*(fb_cs[i, 1] / 10.)**fp[9] * np.log10(fb_cs[i, 0])**3)
        model_params[i, 3] = fp[10] + fp[11]*fb_cs[i,0]**fp[12]
    return model_params

@jit(parallel=True)#, fastmath=True)
def paramet_hayashi_deluxe_v4(fb_cs, fp):
    model_params = np.zeros((len(fb_cs), 4))
    for i in prange(0, len(fb_cs)):
        # can also try setting the first term in f_t free to allow deviation from zero
        # and can allow first term in r_t to deviate from r_vir
        
        model_params[i, 0] = 10**(0. + fp[0]*((fb_cs[i, 1] / 10.)**fp[1]) * np.log10(
            fb_cs[i, 0]) + fp[2]*((fb_cs[i, 1] / 10.)**fp[3]) * np.log10(fb_cs[i, 0])**2.)
        model_params[i, 1] = fb_cs[i, 1]  # just c_s
        model_params[i, 2] = 10**(np.log10(fb_cs[i, 1]) + fp[4]*((fb_cs[i, 1] / 10.)**fp[5]) * np.log10(
            fb_cs[i, 0]) + fp[6]*((fb_cs[i, 1] / 10.)**fp[7]) * np.log10(fb_cs[i, 0])**2. + fp[8]*(fb_cs[i, 1] / 10.)**fp[9] * np.log10(fb_cs[i, 0])**3)
        model_params[i, 3] = fp[10]
    return model_params


# In[67]:


# method for computing the vmax, rmax given a transfer function
# we'll assume the transfer function depends on r, f_b, and c_s
# the initial density profile is an NFW profile, and we now need to write a function for the NFW, which depends
# only on c_s
# need to get it to agree with our profile for a simulated halo in the model units

# once we have this profile function, called NFWrho or something, we can write as mass profile function
# then rmax can be calculated from minimizing this

#everything that we need for the penarrubia model of vmax, rmax
#(mu,eta) = (0.4, 0.3) for vmax
#(mu,eta) = (-0.3, 0.4) for rmax
def g_v_r(x,mu,eta):
    return 2**mu * x**eta / (1+x)**mu


def NFWrho(r, cs):  # in units where Rvir=1, so r_s = 1/c_s
    rhos = (4*np.pi * cs**-3 * NFWf(cs))**-1  # this factor drops out...
    return rhos / ((r*cs)*(1. + r*cs)**2)


def NFWmass(r, cs):
    return 4*np.pi*quad(lambda x: x**2 * NFWrho(x, cs), 0, r)[0]


def hayashiMass(r, fb, cs):
    return 4*np.pi*quad(lambda x: x**2 * NFWrho(x, cs)*strip_mod_hayashi(x*cs, fb), 0, r)[0]


def penarrubiaMass(r, fb, cs):
    return 4*np.pi*quad(lambda x: x**2 * pen_new_prof(x*cs, fb, cs), 0, r)[0]


def rte(fb):
    return 10**(1.02 + 1.38*np.log10(fb) + 0.37*(np.log10(fb))**2)


def ft(fb):
    return 10**(-0.007 + 0.35*np.log10(fb) + 0.39*(np.log10(fb))**2 + 0.23*(np.log10(fb))**3)


def strip_mod_hayashi(r, fb):
    return ft(fb) / (1. + (r/rte(fb))**3)

from scipy.optimize import root_scalar
def hayashi_vm_rm(fb, cs, normed=True):
    #rmax = root(lambda x: 4*np.pi*x**3 * strip_mod_hayashi(x*cs, fb)
    #            * NFWrho(x, cs) - hayashiMass(x, fb, cs), g_v_r(fb, -0.3,0.4)*2.163/cs).x[0]
    rmax = root_scalar(lambda x: 4*np.pi*x**3 * strip_mod_hayashi(x*cs, fb)
                * NFWrho(x, cs) - hayashiMass(x, fb, cs), bracket=(10**-3, 1.2)).root
    vmax = np.sqrt(G*hayashiMass(rmax, fb, cs) / rmax)
    # need these as ratios...
    if(normed):
        # root(lambda x: x**3 * NFWrho(x, cs) - NFWmass(x, cs), 2.163/cs).x[0]
        rmax0 = 2.163/cs
        # np.sqrt(G*NFWmass(rmax0, cs) / rmax0) #NFW for IC
        vmax0 = 0.465*np.sqrt(cs/NFWf(cs))
        return rmax/rmax0, vmax/vmax0
    else:
        return rmax, vmax


def modelMass(fit_parms, model_param_func, rho_model, r, fb, cs):
    model_params = model_param_func(np.column_stack((fb, cs)), fit_parms)
    return 4*np.pi*quad(lambda x: x**2 * rho_model(x*cs, model_params[0])*NFWrho(x, cs), 0, r)[0]

# this is for our model
# need to pass the model, parameterization, and parameters
def model_vm_rm(fit_parms, model_param_func, rho_model, fb, cs, normed=True):
    model_params = model_param_func(np.column_stack((fb, cs)), fit_parms)
    #rmax = root(lambda x: 4*np.pi*x**3 * rho_model(x*cs, model_params[0]) * NFWrho(
    #    x, cs) - modelMass(fit_parms, model_param_func, rho_model, x, fb, cs), g_v_r(fb, -0.3,0.4)* 2.163/cs).x[0]
    rmax = root_scalar(lambda x: 4*np.pi*x**3 * rho_model(x*cs, model_params[0]) * NFWrho(
        x, cs) - modelMass(fit_parms, model_param_func, rho_model, x, fb, cs), bracket=(10**-10, 1.2)).root
    vmax = np.sqrt(G*modelMass(fit_parms, model_param_func, rho_model, rmax, fb, cs) / rmax)
    if(normed):
        rmax0 = 2.163/cs
        vmax0 = 0.465*np.sqrt(cs/NFWf(cs))
        return rmax/rmax0, vmax/vmax0
    else:
        return rmax, vmax


# In[68]:


model_vmax = np.zeros(len(max_mat))
model_rmax = np.zeros(len(max_mat))
for i in prange(0,len(max_mat)):
    if i % 100 == 0:
        print(i)
    model_rmax[i], model_vmax[i] = model_vm_rm(fp_v52_avgnfw_no1Rv_all, paramet_hayashi_deluxe_v52, hayashi_deluxe, max_mat[i,4], max_mat[i,1], normed=True) 

    #model_rmax = model_rmax / vm_rm_arr[:,1]
#model_vmax = model_vmax / vm_rm_arr[:,0]

# so our model values are measured relative to t=0
# our data values are also measured relative to t=0
# whereas hayashi are measured relative to NFW
# and penarrubia isn't measured relative to anything

# my final concern is that the second parameter in the model is pretty low, so you would think that would justify
# throwing it out... however, we are using consistent power-law shapes for all of them, so that is nice


# In[69]:


#now we need to get these values for every single thing in our dataset

@jit(parallel=True)
def get_hayashi_vr(normed=True):
    hayashi_vmax = np.zeros(len(max_mat))
    hayashi_rmax = np.zeros(len(max_mat))
    for i in prange(0,len(max_mat)):
        if i % 1000 == 0:
            print(i)
        hayashi_rmax[i], hayashi_vmax[i] = hayashi_vm_rm(max_mat[i,4], max_mat[i,1], normed)
    return hayashi_rmax, hayashi_vmax
        
hayashi_rmax, hayashi_vmax = get_hayashi_vr(normed=True)

# this seems to only be using one core... how do we make it use all of them?


# In[70]:


### FIGURE NO LONGER INCLUDED IN PAPER ###

#generate two of our pdfs. Penarrubia vs. ours for vmax/rmax
msk = np.logical_and(np.logical_and(np.logical_and(max_mat[:,5] != 1,~np.isclose(model_rmax,0.)),max_mat[:,6] != 1),~np.isclose(hayashi_rmax,0.))
print(np.sum(msk))
# not sure how to interpret the cs dependence since hayashi calibrated with cs=10 and penarrubia with cs=23
# why is the agreement very bad, and what does this say for us? I should make sure the penarrubia method is correct
#csn = 9
#fbv = 0.45
#fbd = 0.05
#etn = 1
#msk = np.logical_and(msk, max_mat[:,1] == cs_vals[csn])
#msk = np.logical_and(np.logical_and(msk, max_mat[:,4] >= fbv-fbd), max_mat[:,4] <= fbv+fbd)
#print(cs_vals[csn])

# figure out why changing vmax/rmax preds for Hayashi changes the way the other curves look
# it is because with the old root finder we got 8588
# but with the new root finder that uses secant method, we get 8683 nonzero

# need to fix the location of the legend
# what happens to our results if we use the second root finder for our results as well?

vmax_errors_pen = (g_v_r(max_mat[msk,4], 0.4, 0.3) - max_mat[msk,5]) / max_mat[msk,5]
rmax_errors_pen = (g_v_r(max_mat[msk,4], -0.3,0.4) - max_mat[msk,6]) / max_mat[msk,6]

vmax_errors_hay = (hayashi_vmax[msk] - max_mat[msk,5]) / max_mat[msk,5]
rmax_errors_hay = (hayashi_rmax[msk] - max_mat[msk,6]) / max_mat[msk,6]

vmax_errors_mod = (model_vmax[msk] - max_mat[msk,5]) / max_mat[msk,5]
rmax_errors_mod = (model_rmax[msk] - max_mat[msk,6]) / max_mat[msk,6]

#the couple bad ones are where the model is basically zero, so we throw those out

print(int(np.sqrt(len(rmax_errors_hay))))
print(np.min(rmax_errors_hay),np.max(rmax_errors_hay))

bin_edges_v = (-0.4,0.2)
bin_edges_r = (-1,0.5)

fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(14,5))
ax[0].yaxis.set_ticks_position('both')
ax[0].xaxis.set_ticks_position('both')
ax[0].tick_params(axis='both', which='minor', colors='black', width=1.0, length=2.0)
ax[1].yaxis.set_ticks_position('both')
ax[1].xaxis.set_ticks_position('both')
ax[1].tick_params(axis='both', which='minor', colors='black', width=1.0, length=2.0)
ax[0].xaxis.set_minor_locator(MultipleLocator(0.02))
ax[0].yaxis.set_minor_locator(MultipleLocator(1))
ax[1].xaxis.set_minor_locator(MultipleLocator(0.05))
ax[1].yaxis.set_minor_locator(MultipleLocator(1))

ax[0].hist(vmax_errors_mod, bins=int(np.sqrt(len(vmax_errors_mod))), normed=True, color="b", linestyle='solid', histtype="step", linewidth=2., label='This work', range=bin_edges_v);
ax[0].hist(vmax_errors_pen, bins=int(np.sqrt(len(vmax_errors_pen))), normed=True, color="k", linestyle='dashed', histtype="step", linewidth=2., label='Pe{\~ n}arrubia+2010', range=bin_edges_v);
ax[0].hist(vmax_errors_hay, bins=int(np.sqrt(len(vmax_errors_hay))), normed=True, color="r", linestyle='dotted', histtype="step", linewidth=2., label='Hayashi+2003', range=bin_edges_v);
ax[0].axvline(0., color='k')
#ax[0].set_xlabel(r'$[\frac{V_{max}}{V_{max,i}}\textrm{(DASH)} - \frac{V_{max}}{V_{max,i}}\textrm{(Model)}] / \frac{V_{max}}{V_{max,i}}\textrm{(DASH)}$')
ax[0].set_xlabel(r'$\delta V_\mathrm{max} / V_\mathrm{max}$')
ax[0].set_xlim(-0.15, 0.2)
ax[0].set_ylabel(r'PDF')
ax[1].hist(rmax_errors_mod, bins=int(np.sqrt(len(rmax_errors_mod))), normed=True, color="b", linestyle='solid', histtype="step", linewidth=2., label='This work', range=bin_edges_r);
ax[1].hist(rmax_errors_pen, bins=int(np.sqrt(len(rmax_errors_pen))), normed=True, color="k", linestyle='dashed', histtype="step", linewidth=2., label='Pe{\~ n}arrubia+2010', range=bin_edges_r);
ax[1].hist(rmax_errors_hay, bins=int(np.sqrt(len(rmax_errors_hay))), normed=True, color="r", linestyle='dotted', histtype="step", linewidth=2., label='Hayashi+2003', range=bin_edges_r);
#ax[1].set_xlabel(r'$[\frac{R_{max}}{R_{max,i}}\textrm{(DASH)} - \frac{R_{max}}{R_{max,i}}\textrm{(Model)}] / \frac{R_{max}}{R_{max,i}}\textrm{(DASH)}$')
ax[1].set_xlabel(r'$\delta r_\mathrm{max} / r_\mathrm{max}$')
ax[1].axvline(0., color='k')
ax[1].set_xlim(-0.5,0.5)
ax[0].legend(loc=1, frameon=False)

mean = 0; std = 0.02; variance = np.square(std)
x = np.arange(-0.1,0.1,.01)
f = np.exp(-np.square(x-mean)/(2*variance))/(np.sqrt(2*np.pi*variance))
#ax[0].plot(x,f)

mean = -0.0125; std = 0.045; variance = np.square(std)
x = np.arange(-0.1,0.1,.01)
f = np.exp(-np.square(x-mean)/(2*variance))/(np.sqrt(2*np.pi*variance))
#ax[1].plot(x,f)

#plt.savefig(fig_dir/'vmax_rmax_dist.eps', bbox_inches='tight') # may want to look at this as function of c_s?


# In[71]:


from inspect import getmembers, isclass
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def rasterize_and_save(fname, rasterize_list=None, fig=None, dpi=None,
                       savefig_kw={}):
    """Save a figure with raster and vector components
    This function lets you specify which objects to rasterize at the export
    stage, rather than within each plotting call. Rasterizing certain
    components of a complex figure can significantly reduce file size.
    Inputs
    ------
    fname : str
        Output filename with extension
    rasterize_list : list (or object)
        List of objects to rasterize (or a single object to rasterize)
    fig : matplotlib figure object
        Defaults to current figure
    dpi : int
        Resolution (dots per inch) for rasterizing
    savefig_kw : dict
        Extra keywords to pass to matplotlib.pyplot.savefig
    If rasterize_list is not specified, then all contour, pcolor, and
    collects objects (e.g., ``scatter, fill_between`` etc) will be
    rasterized
    Note: does not work correctly with round=True in Basemap
    Example
    -------
    Rasterize the contour, pcolor, and scatter plots, but not the line
    >>> import matplotlib.pyplot as plt
    >>> from numpy.random import random
    >>> X, Y, Z = random((9, 9)), random((9, 9)), random((9, 9))
    >>> fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
    >>> cax1 = ax1.contourf(Z)
    >>> cax2 = ax2.scatter(X, Y, s=Z)
    >>> cax3 = ax3.pcolormesh(Z)
    >>> cax4 = ax4.plot(Z[:, 0])
    >>> rasterize_list = [cax1, cax2, cax3]
    >>> rasterize_and_save('out.svg', rasterize_list, fig=fig, dpi=300)
    """

    # Behave like pyplot and act on current figure if no figure is specified
    fig = plt.gcf() if fig is None else fig

    # Need to set_rasterization_zorder in order for rasterizing to work
    zorder = -5  # Somewhat arbitrary, just ensuring less than 0

    if rasterize_list is None:
        # Have a guess at stuff that should be rasterised
        types_to_raster = ['QuadMesh', 'Contour', 'collections']
        rasterize_list = []

        print("""
        No rasterize_list specified, so the following objects will
        be rasterized: """)
        # Get all axes, and then get objects within axes
        for ax in fig.get_axes():
            for item in ax.get_children():
                if any(x in str(item) for x in types_to_raster):
                    rasterize_list.append(item)
        print('\n'.join([str(x) for x in rasterize_list]))
    else:
        # Allow rasterize_list to be input as an object to rasterize
        if type(rasterize_list) != list:
            rasterize_list = [rasterize_list]

    for item in rasterize_list:

        # Whether or not plot is a contour plot is important
        is_contour = (isinstance(item, matplotlib.contour.QuadContourSet) or
                      isinstance(item, matplotlib.tri.TriContourSet))

        # Whether or not collection of lines
        # This is commented as we seldom want to rasterize lines
        # is_lines = isinstance(item, matplotlib.collections.LineCollection)

        # Whether or not current item is list of patches
        all_patch_types = tuple(
            x[1] for x in getmembers(matplotlib.patches, isclass))
        try:
            is_patch_list = isinstance(item[0], all_patch_types)
        except TypeError:
            is_patch_list = False

        # Convert to rasterized mode and then change zorder properties
        if is_contour:
            curr_ax = item.ax.axes
            curr_ax.set_rasterization_zorder(zorder)
            # For contour plots, need to set each part of the contour
            # collection individually
            for contour_level in item.collections:
                contour_level.set_zorder(zorder - 1)
                contour_level.set_rasterized(True)
        elif is_patch_list:
            # For list of patches, need to set zorder for each patch
            for patch in item:
                curr_ax = patch.axes
                curr_ax.set_rasterization_zorder(zorder)
                patch.set_zorder(zorder - 1)
                patch.set_rasterized(True)
        else:
            # For all other objects, we can just do it all at once
            curr_ax = item.axes
            curr_ax.set_rasterization_zorder(zorder)
            item.set_rasterized(True)
            item.set_zorder(zorder - 1)

    # dpi is a savefig keyword argument, but treat it as special since it is
    # important to this function
    if dpi is not None:
        savefig_kw['dpi'] = dpi

    # Save resulting figure
    fig.savefig(fname, **savefig_kw)


# In[72]:


### FIGURE INCLUDED IN PAPER ###

from matplotlib.colors import ListedColormap
my_cmap = ListedColormap(sns.cubehelix_palette(256).as_hex())

msk = np.logical_and(np.logical_and(np.logical_and(max_mat[:,5] != 1,~np.isclose(model_rmax,0.)),max_mat[:,6] != 1),~np.isclose(hayashi_rmax,0.))
msk = np.logical_and(msk, (model_rmax - max_mat[:,6]) / max_mat[:,6] > -0.25)

fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(15.5,6))
ax[0].yaxis.set_ticks_position('both')
ax[0].xaxis.set_ticks_position('both')
#ax[0].loglog()
ax[0].tick_params(axis='both', which='minor', colors='black', width=1.0, length=2.0)
ax[1].yaxis.set_ticks_position('both')
ax[1].xaxis.set_ticks_position('both')
#ax[1].loglog()
ax[1].tick_params(axis='both', which='minor', colors='black', width=1.0, length=2.0)
ax[0].xaxis.set_minor_locator(MultipleLocator(0.05))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.05))
ax[1].xaxis.set_minor_locator(MultipleLocator(0.05))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.05))
#cs_num = 2
#sc = plt.scatter(max_mat[msk,5][np.where(max_mat[msk,1] ==cs_vals[cs_num])], model_vmax[msk][np.where(max_mat[msk,1] ==cs_vals[cs_num])], c=np.log10(max_mat[msk,1])[np.where(max_mat[msk,1] ==cs_vals[cs_num])], cmap=plt.get_cmap('jet'))
sc1 = ax[0].scatter(np.log10(max_mat[msk,5]), np.log10(hayashi_vmax[msk]), marker='*', color='k', label=r'Hayashi+2003',zorder=-2)
sc2 = ax[0].scatter(np.log10(max_mat[msk,5]), np.log10(g_v_r(max_mat[msk,4], 0.4,0.3)), marker='x', color='gray', label=r'Pe{\~ n}arrubia+2010',zorder=-1,alpha=0.8)
sc3 = ax[0].scatter(np.log10(max_mat[msk,5]), np.log10(model_vmax[msk]), c=np.log10(max_mat[msk,1]), cmap=my_cmap, label=r'This work',zorder=0)
ax[0].set_xlabel(r'$\log\Big[V_\mathrm{max} / V_{\mathrm{max},i} \, (\mathrm{DASH})\Big]$')
ax[0].set_ylabel(r'$\log\Big[V_\mathrm{max} / V_{\mathrm{max},i} \, (\mathrm{Model})\Big]$');
ax[0].legend(frameon=False,loc=2);
ax[0].set_ylim(np.log10(1.5e-1),np.log10(1.1))

leg = ax[0].get_legend()
leg.legendHandles[2].set_color(sns.cubehelix_palette(256)[128])

#ax[0].set_rasterization_zorder(1)
#cbar = plt.colorbar(sc);
#cbar.set_label(r'$\log_{10}(c_s)$');

#fig, ax  = loglogplot()
sc4 = ax[1].scatter(np.log10(max_mat[msk,6]), np.log10(hayashi_rmax[msk]), marker='*', color='k', label=r'Hayashi+2003',zorder=-2)
sc5 = ax[1].scatter(np.log10(max_mat[msk,6]), np.log10(g_v_r(max_mat[msk,4], -0.3,0.4)), marker='x', color='gray', label=r'Pe{\~ n}arrubia+2010',zorder=-1,alpha=0.8)
sc6 = ax[1].scatter(np.log10(max_mat[msk,6]), np.log10(model_rmax[msk]), c=np.log10(max_mat[msk,1]), cmap=my_cmap,zorder=0)
#ax[1].plot([np.min(max_mat[msk,6]),np.max(max_mat[msk,6])],[np.min(max_mat[msk,6]),np.max(max_mat[msk,6])], 'k')
ax[1].set_xlabel(r'$\log\Big[r_\mathrm{max} / r_{\mathrm{max},i} \, (\mathrm{DASH})\Big]$')
ax[1].set_ylabel(r'$\log\Big[r_\mathrm{max} / r_{\mathrm{max},i} \, (\mathrm{Model})\Big]$');
ax[1].set_rasterization_zorder(1)
#cbar = plt.colorbar(sc);
#cbar.set_label(r'$\log_{10}(c_s)$');

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.12, 0.03, 0.76])
fig.colorbar(sc6, cax=cbar_ax, label=r'$\log(c_\mathrm{s})$')

for i in range(0,2):
    lims = [
        np.min([ax[i].get_xlim(), ax[i].get_ylim()]),  # min of both axes
        np.max([ax[i].get_xlim(), ax[i].get_ylim()]),  # max of both axes
    ]
    ax[i].plot(lims, lims, 'k')
    ax[i].set_xlim(lims)
    ax[i].set_ylim(lims)

rasterize_list = [sc1,sc2,sc3,sc4,sc5,sc6]

#plt.savefig(fig_dir / 'vmax_rmax_scatter.eps', bbox_inches='tight', dpi=300)
#rasterize_and_save(fig_dir/'vmax_rmax_scatter.pdf', rasterize_list, dpi=300, savefig_kw=dict(bbox_inches='tight'))


# In[73]:


### FIGURE INCLUDED IN PAPER ###

# now, I think we want to make this plot binned by c_s and see how things look for different c_s values
from matplotlib.colors import ListedColormap
my_cmap = ListedColormap(sns.cubehelix_palette(256).as_hex())

msk = max_mat[:,6] < 1.47*max_mat[:,4]**(0.43)

fb_bins = 30
fb_vals = np.logspace(-3, 0, fb_bins)

# need to overplot some bin averages to see what's going on since it looks like Penarrubia does a good job
# then can also overplot the Hayashi model predictions using my other approach
bins = 10
median_vmax = np.zeros(bins); vmax_quants = np.zeros((bins,2))
median_rmax = np.zeros(bins); rmax_quants = np.zeros((bins,2))
n_per_bin = np.zeros(bins)
frac_diff_v = np.zeros(bins); frac_diff_r = np.zeros(bins)
fb_bin_edges = np.logspace(-3,0,bins+1)
fb_bin_centers = 10**((np.log10(fb_bin_edges[1:]) + np.log10(fb_bin_edges[:-1]))*0.5)
for i in range(0,bins):
    sel = np.logical_and(max_mat[:,4] > fb_bin_edges[i], max_mat[:,4] <= fb_bin_edges[i+1])
    sel = np.logical_and(sel, msk)
    n_per_bin[i] = np.sum(sel)
    median_vmax[i] = np.median(max_mat[sel,5])
    median_rmax[i] = np.median(max_mat[sel,6])
    vmax_quants[i,:] = np.percentile(max_mat[sel,5],[16,84])
    rmax_quants[i,:] = np.percentile(max_mat[sel,6],[16,84])
    vmax_quants[i,0] = np.log10(median_vmax[i]) - np.log10(vmax_quants[i,0])
    vmax_quants[i,1] = np.log10(vmax_quants[i,1]) - np.log10(median_vmax[i])
    rmax_quants[i,0] = np.log10(median_rmax[i]) - np.log10(rmax_quants[i,0])
    rmax_quants[i,1] = np.log10(rmax_quants[i,1]) - np.log10(median_rmax[i])
    frac_diff_v[i] = (g_v_r(fb_bin_centers[i], 0.4,0.3) - median_vmax[i])/median_vmax[i]
    frac_diff_r[i] = (g_v_r(fb_bin_centers[i], -0.3,0.4) - median_rmax[i])/median_rmax[i]

# above each, we can put a table with N and the fractional between the penarrubia prediction and the median value

# now, can we figure out a way to overplot the Hayashi predictions via the integration method for 23.1 and 10?
# want to do this for the fb_vals, of which there are 30... we can do this for two concentrations

# returns rmax and vmax as 0,1 column
hay_preds_c10 = np.zeros((fb_bins,2))
hay_preds_c25 = np.zeros((fb_bins,2))
model_preds_c5 = np.zeros((fb_bins,2))
model_preds_c10 = np.zeros((fb_bins,2))
model_preds_c25 = np.zeros((fb_bins,2))
for i in range(0,fb_bins):
    hay_preds_c10[i,:] = hayashi_vm_rm(fb_vals[i], 10, normed=True)
    hay_preds_c25[i,:] = hayashi_vm_rm(fb_vals[i], 25, normed=True)
    model_preds_c5[i,:] = model_vm_rm(fp_v52_avgnfw_no1Rv_all, paramet_hayashi_deluxe_v52, hayashi_deluxe, fb_vals[i], 5, normed=True)
    model_preds_c10[i,:] = model_vm_rm(fp_v52_avgnfw_no1Rv_all, paramet_hayashi_deluxe_v52, hayashi_deluxe, fb_vals[i], 10, normed=True)
    model_preds_c25[i,:] = model_vm_rm(fp_v52_avgnfw_no1Rv_all, paramet_hayashi_deluxe_v52, hayashi_deluxe, fb_vals[i], 25, normed=True)
# there is a problem here... why doesn't it show up when we do the comparison plots?
# why do the Hayashi predictions look fine below fb=10**-2, or do they?

# hayashi_vm_rm(max_mat[i,4], max_mat[i,1], normed)

fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(15.5, 6))
ax[0].yaxis.set_ticks_position('both')
ax[0].xaxis.set_ticks_position('both')
ax[0].tick_params(axis='both', which='minor', colors='black', width=1.0, length=2.0)
ax[1].yaxis.set_ticks_position('both')
ax[1].xaxis.set_ticks_position('both')
ax[1].tick_params(axis='both', which='minor', colors='black', width=1.0, length=2.0)
ax[0].xaxis.set_minor_locator(MultipleLocator(0.1))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.05))
ax[1].xaxis.set_minor_locator(MultipleLocator(0.1))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.05))
#ax[0].loglog(); ax[1].loglog() #might change this to regular space, pending Frank...; also want to rasterize


sc1 = ax[0].scatter(np.log10(max_mat[msk,4]), np.log10(max_mat[msk,5]), c=np.log10(max_mat[msk,1]), cmap=my_cmap)
ax[0].plot(np.log10(fb_vals), np.log10(g_v_r(fb_vals, 0.4,0.3)),'lime', linewidth=2, label=r"P10, ``tidal track''")
ax[0].plot(np.log10(fb_vals), np.log10(fb_vals**(1./3.)),'red',linestyle='dashed',linewidth=2, label=r'H03, $V_\mathrm{max}\propto f_\mathrm{b}^{1/3}$')
#for i in range(0,bins):
#    plt.text(fb_bin_centers[i], 1.3, '%d' % n_per_bin[i])
#    plt.text(fb_bin_centers[i], 1.5, '%.2f' % frac_diff_v[i])
#(_, caps, _) = ax[0].errorbar(np.log10(fb_bin_centers), np.log10(median_vmax), yerr=vmax_quants.T, fmt='.', color = 'blue', markersize=10, linewidth=2, capsize=3, capthick=2, zorder=32)
#for cap in caps:
#    cap.set_color('k')
#    cap.set_markeredgewidth(1)
ax[0].plot(np.log10(fb_vals), np.log10(hay_preds_c25[:,1]), color='red', linewidth=2, linestyle='dotted', label=r'H03, using $H_\mathrm{H03}(r|f_\mathrm{b})$')
lm1, = ax[0].plot(np.log10(fb_vals), np.log10(model_preds_c5[:,1]), color='blue', linewidth=2, linestyle='dotted')
lm2, = ax[0].plot(np.log10(fb_vals), np.log10(model_preds_c10[:,1]), color='blue', linewidth=2, linestyle='solid')
lm3, = ax[0].plot(np.log10(fb_vals), np.log10(model_preds_c25[:,1]), color='blue', linewidth=2, linestyle='dashed')
lines = [lm1, lm2, lm3]
legend1 = fig.legend(lines, [r'5',r'10',r'25'], title=r'This work, $c_\mathrm{s}=$', title_fontsize=18, frameon=False, bbox_to_anchor=(0,1), bbox_transform=ax[0].transAxes, loc=2)
#ax[0].add_artist(legend1)
#plt.plot(fb_vals, hay_preds_c10[:,1])
ax[0].set_xlabel(r'$\log\Big[f_b\Big]$')
ax[0].set_ylabel(r'$\log\Big[V_\mathrm{max} / V_{\mathrm{max},i}\Big]$');
#cbar = plt.colorbar(sc);
#cbar.set_label(r'$c_s$');
ax[0].set_ylim(np.log10(9e-2),np.log10(1.1))
ax[0].legend(frameon=False);

#fig, ax  = loglogplot()
sc2 = ax[1].scatter(np.log10(max_mat[msk,4]), np.log10(max_mat[msk,6]), c=np.log10(max_mat[msk,1]), cmap=my_cmap) #by c_s
ax[1].plot(np.log10(fb_vals), np.log10(g_v_r(fb_vals, -0.3,0.4)),'lime', linewidth=2, label=r'P10, $\gamma=1$')
#(_, caps, _) = ax[1].errorbar(np.log10(fb_bin_centers), np.log10(median_rmax), yerr=rmax_quants.T, fmt='.', color = 'blue', markersize=10, linewidth=2, capsize=3, capthick=2, zorder=32)
#for i in range(0,bins):
#    plt.text(fb_bin_centers[i], 1.3, '%d' % n_per_bin[i])
#    plt.text(fb_bin_centers[i], 1.5, '%.2f' % frac_diff_r[i])
ax[1].plot(np.log10(fb_vals), np.log10(hay_preds_c25[:,0]), color='red', linewidth=2, linestyle='dotted', label=r'H03, using $H_\mathrm{H03}(r|f_\mathrm{b})$')
ax[1].plot(np.log10(fb_vals), np.log10(model_preds_c5[:,0]), color='blue', linewidth=2, linestyle='dotted', label=r'This work, $c_\mathrm{s}=5$')
ax[1].plot(np.log10(fb_vals), np.log10(model_preds_c10[:,0]), color='blue', linewidth=2, linestyle='solid', label=r'This work, $c_\mathrm{s}=10$')
ax[1].plot(np.log10(fb_vals), np.log10(model_preds_c25[:,0]), color='blue', linewidth=2, linestyle='dashed', label=r'This work, $c_\mathrm{s}=25$')
ax[1].set_xlabel(r'$\log\Big[f_b\Big]$')
ax[1].set_ylabel(r'$\log\Big[r_\mathrm{max} / r_{\mathrm{max},i}\Big]$');
#cbar = ax[1].colorbar(sc);
#cbar.set_label(r'$c_s$');
ax[1].set_ylim(np.log10(4e-2),np.log10(1.1))
#ax[1].legend(loc=4,frameon=False);

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.12, 0.03, 0.76])
fig.colorbar(sc2, cax=cbar_ax, label=r'$\log(c_\mathrm{s})$')

#plt.plot(fb_vals,1.45*fb_vals**(0.43)) # add this to the mask

rasterize_list = [sc1,sc2]

#rasterize_and_save(fig_dir/'vmax_rmax_vs_fb.pdf', rasterize_list, dpi=300, savefig_kw=dict(bbox_inches='tight'))


# ### Fitting functions for $V_\mathrm{max}$ and $r_\mathrm{max}$

# In[76]:


# calibrate the fitting function against the model instead... we just need data points spaced uniformly in
# f_b and c_s and we can see how well it does with various simplified models
# generate vmax, rmax from a matrix of f_b and c_s
n_fb = 80
n_cs = 50
fb_vals = np.logspace(-3.5,0,n_fb)
logcs_vals = np.log10(np.logspace(np.log10(cs_vals[0]),np.log10(cs_vals[-1]),n_cs))
fb_logcs = np.zeros((n_fb*n_cs,2))
for i in range(0,n_fb):
    for j in range(0,n_cs):
        fb_logcs[i*n_cs + j,0] = fb_vals[i]
        fb_logcs[i*n_cs + j,1] = logcs_vals[j]
# now we have our values, we need to get the rmax, vmax predictions from the model

train_vmax = np.zeros(len(fb_logcs))
train_rmax = np.zeros(len(fb_logcs))
for i in prange(0,len(fb_logcs)):
    if i % 100 == 0:
        print(i)
    train_rmax[i], train_vmax[i] = model_vm_rm(fp_v52_avgnfw_no1Rv_all, paramet_hayashi_deluxe_v52, hayashi_deluxe, fb_logcs[i,0], 10**fb_logcs[i,1], normed=True)


# In[77]:


def vr_fit_func_final(fb_logcs,*p):
    mu = p[0] + p[1] * (10**fb_logcs[:,1])**p[2] * np.log10(fb_logcs[:,0]) + p[3] * (10**fb_logcs[:,1])**p[4]
    eta =p[5] + p[6] * (10**fb_logcs[:,1])**p[7] * np.log10(fb_logcs[:,0]) #+ p[8] * (10**fb_logcs[:,1])**p[9]
    return 2**mu * fb_logcs[:,0]**eta / (1+fb_logcs[:,0])**mu

vm_parm, vm_cv = curve_fit(vr_fit_func_final, fb_logcs, train_vmax, p0=[0.4, 0., 0., 0., 0., 0.3, 0., 0.], bounds=([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],np.inf), maxfev=100000)
rm_parm, rm_cv = curve_fit(vr_fit_func_final, fb_logcs, train_rmax, p0=[-0.3, 0., 0., 0., 0., 0.4, 0., 0.], bounds=([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],np.inf), maxfev=100000)


# In[78]:


vm_parm, rm_parm


# In[79]:


def vr_fit_func_final(fb_logcs,p):
    mu = p[0] + p[1] * (10**fb_logcs[:,1])**p[2] * np.log10(fb_logcs[:,0]) + p[3] * (10**fb_logcs[:,1])**p[4]
    eta =p[5] + p[6] * (10**fb_logcs[:,1])**p[7] * np.log10(fb_logcs[:,0])
    return 2**mu * fb_logcs[:,0]**eta / (1+fb_logcs[:,0])**mu

fb_bins = 30
fb_vals = np.logspace(-3.5, 0, fb_bins)

# TODO: still need to find some way to actually incorporate in the f_b dependence so that the slope changes
# This should be a simple modification without overdoing it

# now a figure that just compares our model to the fitting function to demonstrate how well it does
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(16,4))
ax[0].yaxis.set_ticks_position('both')
ax[0].xaxis.set_ticks_position('both')
ax[0].tick_params(axis='both', which='minor', colors='black', width=1.0, length=2.0)
ax[1].yaxis.set_ticks_position('both')
ax[1].xaxis.set_ticks_position('both')
ax[1].tick_params(axis='both', which='minor', colors='black', width=1.0, length=2.0)
ax[0].xaxis.set_minor_locator(MultipleLocator(0.1))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.05))
ax[1].xaxis.set_minor_locator(MultipleLocator(0.1))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.05))

ax[0].set_xlabel(r'$\log\Big[f_b\Big]$')
ax[0].set_ylabel(r'$\log\Big[V_\mathrm{max} / V_{\mathrm{max},i}\Big]$');
ax[1].set_xlabel(r'$\log\Big[f_b\Big]$')
ax[1].set_ylabel(r'$\log\Big[r_\mathrm{max} / r_{\mathrm{max},i}\Big]$');

csn_ns = [2, 5, 9]
cols = sns.cubehelix_palette(len(csn_ns))

model_preds = np.zeros((fb_bins,2))



for coln,csn in enumerate(csn_ns):
    fitting_vm = vr_fit_func_final(np.column_stack((fb_vals,np.repeat(np.log10(cs_vals[csn]),fb_bins))),vm_parm)
    fitting_rm = vr_fit_func_final(np.column_stack((fb_vals,np.repeat(np.log10(cs_vals[csn]),fb_bins))),rm_parm)
    ax[0].plot(np.log10(fb_vals), np.log10(fitting_vm), color=cols[coln], linewidth=2, linestyle='dashed')
    ax[1].plot(np.log10(fb_vals), np.log10(fitting_rm), color=cols[coln], linewidth=2, linestyle='dashed')
    for i in range(0,fb_bins):
        model_preds[i,:] = model_vm_rm(fp_v52_avgnfw_no1Rv_all, paramet_hayashi_deluxe_v52, hayashi_deluxe, fb_vals[i], cs_vals[csn], normed=True)
    ax[0].plot(np.log10(fb_vals), np.log10(model_preds[:,1]), color=cols[coln], linewidth=2, linestyle='solid', label=r'%.2f' % cs_vals[csn])
    ax[1].plot(np.log10(fb_vals), np.log10(model_preds[:,0]), color=cols[coln], linewidth=2, linestyle='solid')
    res_vm = np.max(np.abs((model_preds[:,1] - fitting_vm) / model_preds[:,1]))
    res_rm = np.max(np.abs((model_preds[:,0] - fitting_rm) / model_preds[:,0]))
    print(res_vm,res_rm)

ax[0].legend(title=r'$c_\mathrm{s}$', frameon=False, title_fontsize=18)
#ax[0].set_title('Solid = from transfer function, Dashed = from fitting function for Vmax/Rmax')
#plt.savefig(fig_dir/'vm_rm_fitting_func.eps', bbox_inches='tight')

# The fitting function looks great!

