# VIE-Net: regressive U-Net for Vegetation Index Estimation
End-to-end non-generative model for NDVI estimate from calibrated RGB images. 



# VIE-Net: THE MODEL
VIE-Net is a convolutional neural network based on a regressive version of the U-net architecture. 
<img width="1572" height="1008" alt="image" src="https://github.com/user-attachments/assets/9cebc83c-9eee-47c9-9ab5-4c1d083b1a7d" />


If you use this model please cite the work as: 
Capparella, Valerio and Nemmi, Eugenio and violino, simona and Costa, Corrado and Figorilli, Simone and Moscovini, Lavinia and Pallottino, Federico and Pane, Catello and Mei, Alessandro and ORTENZI, Luciano, Vie-Net: Regressive U-Net for Vegetation Index Estimation. Available at IEEE ACCESS.
<img width="1280" height="720" alt="graphical_abstract_final" src="https://github.com/user-attachments/assets/61f720a0-af04-4ac8-b76a-b7a9854bcb3b" />

# VIE-Net: main results

<img width="335" height="760" alt="image" src="https://github.com/user-attachments/assets/b8b8a4ef-35e0-4134-b823-b0890a197294" />

from Top to Bottom: Calibrated RGB image (through Menesatti et al. Sensors, 12 (2012), pp. 7063-7079); real NDVI map; predicted NDVI map with VIE-Net model.

<img width="360" height="370" alt="image" src="https://github.com/user-attachments/assets/5fdc4e80-c978-4e7b-86d4-231b08439363" />
NDVI squared Error Heat map between ground-truth and predicted NDVI values.

# PRE-TRAINED WEIGHTS:
You can find the pre-trained weights in a new release: Vie-Net 1.0
https://github.com/MANDRACHE-LAB/VIE-Net/releases/tag/mandrache
