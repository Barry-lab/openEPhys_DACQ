import CombineTrackingData
import ApplyKlustakwikScripts
import argparse
import os

# Input argument handling and help info
parser = argparse.ArgumentParser(description='Apply KlustaKwik and export into Axona format.')
parser.add_argument('path', type=str,
                    help='recording data folder')
args = parser.parse_args()
# Form path to recording file
OpenEphysDataPath = args.path
NWBfilePath = os.path.join(OpenEphysDataPath,'experiment_1.nwb')

# Make sure position data is available
if not os.path.exists(os.path.join(os.path.dirname(NWBfilePath),'PosLogComb.csv')):
    if NWBio.check_if_binary_pos(NWBfilePath):
        _ = NWBio.load_pos(NWBfilePath, savecsv=True, postprocess=True)
    else:
        CombineTrackingData.combdata(NWBfilePath)

ApplyKlustakwikScripts.cluster_all_spikes_NWB(NWBfilePath)