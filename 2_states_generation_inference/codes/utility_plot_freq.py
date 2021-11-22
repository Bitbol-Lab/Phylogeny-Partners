import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import cython_code.analyse_sequence as an
import matplotlib.colors as colors
import networkx as nx
import matplotlib.lines as mlines
from . import utility_plot as uti

def func(x, a, b):
    return a + b*x

def without_diag(f):
    lx, ly = [], []
    a = np.array([i for i in range(f.shape[0])])
    for i in range(f.shape[2]):
        lx.extend([i]*(f.shape[2]-1))
        ly.extend(a[:i])
        ly.extend(a[i+1:])
    return lx, ly

def max_min_freq(l_Fij_ref, l_Fij_calc, one_freq):
    if not one_freq:
        lx, ly = without_diag(l_Fij_ref[0])
        max_f = max(np.max(l_Fij_ref[:,lx,:,ly,:]), np.max(l_Fij_calc[:,lx,:,ly,:]))
        min_f = min(np.min(l_Fij_ref[:,lx,:,ly,:]), np.min(l_Fij_calc[:,lx,:,ly,:]))
    else:
        max_f = max(np.max(l_Fij_ref), np.max(l_Fij_calc))
        min_f = min(np.min(l_Fij_ref), np.min(l_Fij_calc))
    return max_f, min_f

def polt_two_l_freq(l_Fij_ref, l_Fij_calc, l_axes_title, l_title=None, width="article", one_freq=False):
    figsize = uti.set_size(width,subplots=(1,len(l_Fij_calc)))
    fig, ax =  plt.subplots(figsize=figsize, ncols=len(l_Fij_calc), nrows=1)
    xmax, xmin = max_min_freq(l_Fij_ref, l_Fij_calc, one_freq)
    ext = [xmin, xmax, xmin, xmax]
    l_hh = []
    for i in range(ax.shape[0]):
        if one_freq:
            l_hh.append(an.image_plot_freq_simple(l_Fij_ref[i].flatten(), l_Fij_calc[i].flatten(), val=100, xmin=xmin, xmax=xmax))
        else:
            l_hh.append(an.image_plot_freq(l_Fij_ref[i], l_Fij_calc[i], val=100, xmin=xmin, xmax=xmax))
    norm = colors.LogNorm(vmin=0.9, vmax=np.max(l_hh))
            
    for i in range(ax.shape[0]):
        r,_ = pearsonr(l_Fij_ref[i].flatten(), l_Fij_calc[i].flatten())
        popt, pcov = curve_fit(func, l_Fij_ref[i].flatten(), l_Fij_calc[i].flatten())
           
        pcm = ax[i].imshow(l_hh[i].T,
                      norm=norm,
                      extent=ext,
                      aspect="auto",
                      origin='lower')
        
        ax[i].plot(ext[0:2], ext[2:], 'g--', linewidth=1)
        
        if l_title is not None:
            print(l_title[i] +"\n a+b*x:" + str(np.round(popt,2)) + "\n pearson = " + str(np.round(r,4)))
            ax[i].set_title(l_title[i])
        if i==0:
            ax[i].set_ylabel(l_axes_title[1])
            ax[i].set_xlabel(l_axes_title[0])
            ax[i].xaxis.set_label_coords(1, -0.2)
        if i!=0:
            ax[i].axes.yaxis.set_ticklabels([])
    cbar = fig.colorbar(pcm,
                    ax=ax.flatten(), label="Counts")
    
    return l_hh

def polt_two_l_freq_contact(l_Fij_ref, l_Fij_calc, graph, cmaps, l_title=None, l_axes_title=None, width="article", x_min=0, x_max=1):
    figsize = uti.set_size(width,subplots=(1,len(l_Fij_calc))) 
    fig, ax =  plt.subplots(figsize=figsize, ncols=len(l_Fij_calc), nrows=1)
    x = np.linspace(x_min,x_max)
    for n in range(ax.shape[0]):
        Fij_calc = l_Fij_calc[n]
        Fij_ref = l_Fij_ref[n]
        r,_ = pearsonr(Fij_ref.flatten(), Fij_calc.flatten())
        popt, pcov = curve_fit(func, Fij_ref.flatten(), Fij_calc.flatten())
        
        l_Fij_contact_ref, l_Fij_no_contact_ref, l_Fij_contact_calc, l_Fij_no_contact_calc = [], [], [], []
        for i in range(Fij_calc.shape[0]):
            for j in range(i, Fij_calc.shape[0]):
                if (i,j) in graph.edges():
                    l_Fij_contact_ref.append(Fij_ref[i,:,j,:].flatten())
                    l_Fij_contact_calc.append(Fij_calc[i,:,j,:].flatten())
                else:
                    l_Fij_no_contact_ref.append(Fij_ref[i,:,j,:].flatten())
                    l_Fij_no_contact_calc.append(Fij_calc[i,:,j,:].flatten())
              
        ax[n].plot(l_Fij_no_contact_ref, l_Fij_no_contact_calc, ls="", marker = ".", markersize=0.1, color=cmaps[1])
        ax[n].plot(l_Fij_contact_ref, l_Fij_contact_calc, ls="", marker = ".", markersize=0.1, color=cmaps[0])
        ax[n].plot(x, x, color="black", lw=0.8)
        
        if l_title is not None:
            ax[n].set_title(l_title[n] +"\n a+b*x:" + str(np.round(popt,2)) + "\n pearson = " + str(np.round(r,4)))
        if n==0:
            ax[n].set_xlabel("Frequences two body in original data")
            ax[n].set_ylabel("Frequences two body in generated data")
            Lhandles = []
            Lhandles.append(mlines.Line2D([],[], ls="", marker = ".", color=cmaps[0], label = "contact"))
            Lhandles.append(mlines.Line2D([],[], ls="", marker = ".", color=cmaps[1], label = "no contact"))
            ax[n].legend(handles=Lhandles)
        if n!=0:
            ax[n].axes.yaxis.set_ticklabels([])
    #fig.savefig("figures/2_body_frequencies.svg",bbox_inches = "tight")