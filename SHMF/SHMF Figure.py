#!/usr/bin/env python
# coding: utf-8

# In[59]:


#########################################################################

# Program that plots SHMF and SHVF, compare model and Bolshoi

# Arthur Fangzhou Jiang 2019 Hebrew University

######################## set up the environment #########################

import numpy as np
import sys

import matplotlib as mpl # must import before pyplot
mpl.use('TkAgg')         # use the 'TkAgg' backend for plots
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 14
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
import seaborn as sns
from useful_functions.plotter import plot, loglogplot

sns.set_style("ticks")
sns.set_context("paper",font_scale=2.0)
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
plt.rc('text', usetex=True)

############################## user input ###############################

#---data
file1 = './SHMF_evolved_Bolshoi.txt' 
file2 = './SHVF_evolved_Bolshoi.txt' 
file3 = './SHMF_evolved_fiducial.txt'
file4 = './SHVF_evolved_fiducial.txt'
file5 = './SHMF_evolved_NoDisruption.txt'
file6 = './SHVF_evolved_NoDisruption.txt'

#---output control
lw=2
size=50
edgewidth = 1
alpha_symbol = 0.5
outfig1 = './FIGURE/shmf_shvf_evolved.pdf'

############################### compute #################################

#--- 
print('>>> ... load ...')
lgmM_Bols,lgdNdlgmM_Bols = np.genfromtxt(file1,skip_header=1,
    delimiter='\t',usecols=(0,1),unpack=True)
lgvV_Bols,lgdNdlgvV_Bols = np.genfromtxt(file2,skip_header=1,
    delimiter='\t',usecols=(0,1),unpack=True)
#
lgmM_fid,lgdNdlgmM_fid = np.genfromtxt(file3,skip_header=1,
    delimiter='\t',usecols=(0,1),unpack=True)
lgvV_fid,lgdNdlgvV_fid = np.genfromtxt(file4,skip_header=1,
    delimiter='\t',usecols=(0,1),unpack=True)
#
lgmM_NoDis,lgdNdlgmM_NoDis = np.genfromtxt(file5,skip_header=1,
    delimiter='\t',usecols=(0,1),unpack=True)
lgvV_NoDis,lgdNdlgvV_NoDis = np.genfromtxt(file6,skip_header=1,
    delimiter='\t',usecols=(0,1),unpack=True)

########################### diagnostic plots ############################

print('>>> plot ...')
# close all previous figure windows
plt.close('all')
  
#------------------------------------------------------------------------

#---set up the figure window
fig1 = plt.figure(figsize=(13,4), dpi=100, facecolor='w', edgecolor='k') 
fig1.subplots_adjust(left=0.08, right=0.98,
    bottom=0.12, top=0.98,hspace=0.2, wspace=0.25)
gs = gridspec.GridSpec(1, 2) 
fig1.suptitle(r' ')

#---
ax = fig1.add_subplot(gs[0,0])
ax.set_xlim(-4.,0.)
ax.set_ylim(-1.,3.) 
ax.set_xlabel(r'$\log \Big[m/M_\mathrm{host}\Big]$',fontsize=18)
ax.set_ylabel(r'$\log [\mathrm{d}N/\mathrm{d}\log(m/M_\mathrm{host})]$',
    fontsize=18)
ax.set_title(r'')
# scale
#ax.set_xscale('log')
#ax.set_yscale('log')
# grid
#ax.grid(which='minor', alpha=0.2)                                                
#ax.grid(which='major', alpha=0.4)
# tick and tick label positions
# start, end = ax.get_xlim()
# major_ticks = np.arange(start, end, 0.5)
# minor_ticks = np.arange(start, end, 0.1)
# ax.set_xticks(major_ticks)
# ax.set_xticks(minor_ticks,minor=True)
start, end = ax.get_ylim()
major_ticks = np.arange(start, end, 1.)
minor_ticks = np.arange(start, end, 0.2)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks,minor=True)
start, end = ax.get_xlim()
major_ticks = np.arange(start, end, 1.)
minor_ticks = np.arange(start, end, 0.2)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks,minor=True)
# tick length
#ax.tick_params('both',direction='in',top='on',right='on',length=5,
#    width=1,which='major',zorder=301)
#ax.tick_params('both',direction='in',top='on',right='on',length=2,
#    width=1,which='minor',zorder=301)
ax.yaxis.set_ticks_position('both')                                   
ax.xaxis.set_ticks_position('both')                                   
ax.tick_params(axis='both', which='minor', colors='black', width=1.0, length=2.0)
# plot
ax.scatter(lgmM_Bols,lgdNdlgmM_Bols,marker='o',s=size,
    facecolor='r',edgecolor='k',linewidth=edgewidth,
    alpha=alpha_symbol,rasterized=False,label=r'Bolshoi')
ax.plot(lgmM_fid,lgdNdlgmM_fid,lw=lw,color='k',label=r'Model')
ax.plot(lgmM_NoDis,lgdNdlgmM_NoDis,lw=lw,ls='--',color='k',
    label=r'Model (No disruption)')
ax.arrow(-2, 0.6, 0, np.log10(2.), head_width=0.07, head_length=0.1, fc='blue', ec='blue', length_includes_head=True, zorder=32)
ax.arrow(-1.6, 0.255, 0, np.log10(2.), head_width=0.07, head_length=0.1, fc='blue', ec='blue', length_includes_head=True, zorder=32)
ax.arrow(-2.4, 0.94, 0, np.log10(2.), head_width=0.07, head_length=0.1, fc='blue', ec='blue', length_includes_head=True, zorder=32)
ax.scatter( [2] ,[0.7], c='blue',marker=r'$\uparrow$',s=200, label=r'Factor of 2' )
# reference lines
# ...
# annotations
#ax.text(0.2,0.36,r'',
#    color='k',fontsize=16,ha='left',va='bottom',
#    transform=ax.transAxes,rotation=45)
# legend
#ax.legend(loc='best',fontsize=12,frameon=True)
ax.legend(loc='best',fontsize=12,frameon=False)

#---
ax = fig1.add_subplot(gs[0,1])
ax.set_xlim(-1.1,0.)
ax.set_ylim(-1.,3.) 
ax.set_xlabel(r'$\log \Big[V_\mathrm{max}/V_\mathrm{vir,h}\Big]$',fontsize=18)
ax.set_ylabel(r'$\log [\mathrm{d}N/\mathrm{d}\log(V_\mathrm{max}/V_\mathrm{vir,h})]$',
    fontsize=18)
ax.set_title(r'')
# scale
#ax.set_xscale('log')
#ax.set_yscale('log')
# grid
#ax.grid(which='minor', alpha=0.2)                                                
#ax.grid(which='major', alpha=0.4)
# tick and tick label positions
# start, end = ax.get_xlim()
# major_ticks = np.arange(start, end, 0.5)
# minor_ticks = np.arange(start, end, 0.1)
# ax.set_xticks(major_ticks)
# ax.set_xticks(minor_ticks,minor=True)
start, end = ax.get_ylim()
major_ticks = np.arange(start, end, 1.)
minor_ticks = np.arange(start, end, 0.2)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks,minor=True)
start, end = ax.get_xlim()
major_ticks = np.array([-1.,-0.5,0.])
minor_ticks = np.arange(start, end, 0.1)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks,minor=True)
# tick length
#ax.tick_params('both',direction='in',top='on',right='on',length=5,
#    width=1,which='major',zorder=301)
#ax.tick_params('both',direction='in',top='on',right='on',length=2,
#    width=1,which='minor',zorder=301)
ax.yaxis.set_ticks_position('both')                                   
ax.xaxis.set_ticks_position('both')                                   
ax.tick_params(axis='both', which='minor', colors='black', width=1.0, length=2.0)
# plot
ax.scatter(lgvV_Bols,lgdNdlgvV_Bols,marker='o',s=size,
    facecolor='r',edgecolor='k',linewidth=edgewidth,
    alpha=alpha_symbol,rasterized=False,label=r'Bolshoi')
ax.plot(lgvV_fid,lgdNdlgvV_fid,lw=lw,color='k',label=r'Model')
ax.plot(lgvV_NoDis,lgdNdlgvV_NoDis,lw=lw,ls='--',color='k',
    label=r'Model (No disruption)')
ax.arrow(-0.54, 1.2, 0, np.log10(2.), head_width=0.02, head_length=0.1, fc='blue', ec='blue', length_includes_head=True, zorder=32)
ax.arrow(-0.4, 0.83, 0, np.log10(2.), head_width=0.02, head_length=0.1, fc='blue', ec='blue', length_includes_head=True, zorder=32)
ax.arrow(-0.68, 1.62, 0, np.log10(2.), head_width=0.02, head_length=0.1, fc='blue', ec='blue', length_includes_head=True, zorder=32)
ax.scatter( [2] ,[0.7], c='blue',marker=r'$\uparrow$',s=200, label=r'Factor of 2' )
# reference lines
# ...
# annotations
#ax.text(0.2,0.36,r'',
#    color='k',fontsize=16,ha='left',va='bottom',
#    transform=ax.transAxes,rotation=45)
# legend
#ax.legend(loc='best',fontsize=12,frameon=False)

#---save figure
plt.savefig(outfig1,bbox_inches='tight',dpi=300)
fig1.canvas.manager.window.attributes('-topmost', 1)
plt.get_current_fig_manager().window.wm_geometry('+50+50')
fig1.show()


# In[ ]:




