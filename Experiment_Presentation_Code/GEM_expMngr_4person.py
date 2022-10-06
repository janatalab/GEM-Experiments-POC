'''
Presets for use in 4-player GEM pilot experiment.

This experiment involves four tappers with an adaptive metronome to extend the
findings of Fairhurst, Janata, and Keller (2012) to group settings.

    IV: alpha value
    DV: individual std asynchronies, group std asynchs, subjective ratings
    Instructions: Listen to first 2 metronome tones and then synchronize.
    Try to maintain the initial tempo of the metronome.
    Tap with index finger of dominant hand. Look at own index finger, not other players.

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

rootpath = "/Users/" + os.environ['USER'] + "/Documents/Arduino/"

presets = {
    "serial": {"port": get_metronome_port(), "baud_rate": 115200, "timeout": 5},
    "filename": "GEM_4playerData",
    "data_dir": "/Users/" + os.environ['USER'] +        "/Desktop/GEM_data/4person_GEM_pilotData/",
    "hfile": rootpath + "GEM/GEM/GEMConstants.h",
    "tappers_requested": 4,
    "metronome_alpha": [0, 0.35, 0.7, 1],
    "metronome_tempo": 120.0, #units: beats-per-minute
    "repeats": 6,
    "windows": 60, # ~30 sec rounds 
    "audio_feedback": ["hear_metronome"],
    "metronome_heuristic": ["average"]
}

if __name__ == "__main__":

    g = GEMGUI(presets)
    g.mainloop()
