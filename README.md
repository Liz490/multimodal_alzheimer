# ADLM ADNI

ADNI is a rich dataset designed to track Alzheimers and potential Alzheimers patients across the US. 3D Brain MRI scans as well as protein-indicating PET scans are used to measure the biological state of the patient in addition to genetic testing, general demographic information, blood biomarkers and cognitive tests.

Much research focuses on creating the best possible model with only one source of data, often leaving a lot of useful information unnecessarily on the table. With so much information from many different sources available, we are interested in investigating how best to combine these pools of information to achieve the best possible classification of Alzheimers disease state. Focus will be on how to optimally combine different modalities, taking care not to encode the same information multiple times while allowing correlations and interactions between data sources to be learned.

# Setup

## Environment
`conda env create -f environment.yml`

## Automatic ssh authentification
Add to your ssh config (~/.ssh/config):
```
Host adlm 
  HostName 131.159.110.3 
  User <YOUR_USERNAME>
```

Connect to the AIMED VPN.
Add your public key to authkeys on the server:
`you@your_computer:~$ ssh-copy-id -i ~/.ssh/<YOUR_PUBLIC_KEY_FILE> adlm`

Now you can login with `ssh adlm` if you are connected to the VPN and use the download script for tabular data.

# Usage

After setting up automatic ssh authentification you can download tabular data by running 
`./download_tabular.sh`
