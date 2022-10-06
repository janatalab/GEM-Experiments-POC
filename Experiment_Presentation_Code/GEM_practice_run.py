'''
Practice run presets used in GEM experiments.

Authors: Lauren Fink, Scottie Alexander, Petr Janata
Contact: pjanata@ucdavis.edu
Repository link: https://github.com/janatalab/GEM
'''

import os, sys, re

# Deal with adding the requisite GEM GUI modules to the path
if not os.environ.get('GEMROOT', None):
    # Try to get the GEM path from this module's path.
    p = re.compile('.*/GEM/')
    m = p.match(os.path.join(os.path.abspath(os.path.curdir),__file__))

    if m:
        os.environ['GEMROOT'] = m.group(0)

sys.path.append(os.path.join(os.environ['GEMROOT'],'GUI'))

# Finish the GEM imports
from GEMGUI import GEMGUI
from GEMIO import get_metronome_port

# Define path to Arduino directory
rootpath = "/Users/" + os.environ['USER'] + "/Documents/Arduino/"

# Define experimental presets
presets = {
    # metronome serial port info
    "serial": {"port": get_metronome_port(), "baud_rate": 115200, "timeout": 5},

    # beginning of output file string for output data files
    "filename": "GEM_practice",

    # directory for output data
    "data_dir": "/Users/" + os.environ['USER'] +        "/Desktop/GEM_data/practice_runs/",

    # path to GEMConstants.h
    "hfile": rootpath + "GEM/GEM/GEMConstants.h",

    # number of players in the experiment. NB: all 4 tapper Arduinos can remain attached to metronome Arduino
    "tappers_requested": 4,

    # metronome adaptivity levels to be used
    "metronome_alpha": 0,

    # tempo of the metronome; unit: beats-per-minute
    "metronome_tempo": 120.0,

    # number of repetitions for each alpha value
    "repeats": 3,

    # number of metronome clicks
    "windows": 26,

    # audio feedback condition; NB: at present, only "hear_metronome" available.
    # Future releases will allow for all variations on hearing self, metronome,
    # and others in the experiment.
    "audio_feedback": ["hear_metronome"],

    # algorithm used in adapting metronome. NB: at present, only "average" is
    # available. Future releases will incorporate additional heurstic options.
    "metronome_heuristic": ["average"]
}


# Run the experiment through the GUI
if __name__ == "__main__":

    g = GEMGUI(presets)
    g.mainloop()
