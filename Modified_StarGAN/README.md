# Voice-Conversion-GAN
Voice Conversion using Modified StarGAN (PyTorch Implementation). Architecture of the Modifeid StarGAN is as follows:
<p align="center">
  <img src="./figures/cycleGAN.png" width="100%">
</p>

## Dependencies

* Python
* Numpy 
* PyTorch 
* LibROSA 
* PyWorld



## Usage

### Download Dataset

Download and unzip [VCC2018](https://datashare.ed.ac.uk/handle/10283/3061) into google drive.



### Preprocessing for Training



### Train Model


For example, to train CycleGAN model for voice conversion between ``SF1`` and ``TF2``:

```bash
$python train.py 
```                                                

