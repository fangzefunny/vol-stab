import numpy as np 
import seaborn as sns 

class viz:
    '''Define the default visualize configure
    '''
    Blue    = np.array([ 46, 107, 149]) / 255
    Green   = np.array([  8, 154, 133]) / 255
    Red     = np.array([199, 111, 132]) / 255
    Yellow  = np.array([220, 175, 106]) / 255
    Purple  = np.array([108,  92, 231]) / 255
    Palette = [Blue, Red, Green, Yellow, Purple]
    Greens  = [np.array([  8, 154, 133]) / 255, 
               np.array([118, 193, 202]) / 255] 
    dpi     = 200
    sfz, mfz, lfz = 11, 13, 16
    lw, mz  = 2.5, 6.5
    figz    = 4

    @staticmethod
    def get_style(): 
        # Larger scale for plots in notebooks
        sns.set_context('talk')
        sns.set_style("ticks", {'axes.grid': False})
