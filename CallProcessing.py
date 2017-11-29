import CombineTrackingData as combPos
import ApplyKlustakwikScripts as AKS
import sys

dataFileName = str(sys.argv[1])

combPos.combdata(dataFileName)
AKS.cluster_all_spikes_Kwik(dataFileName)