## Emotion-guided Controller for DECA models

## Description

The Emotion-guided Controller for DECA models (ECD) is a project aimed at developing a system that leverages emotions to manipulate facial expressions in DECA sequences. DECA (Detailed Expression Capture and Animation) is a state-of-the-art facial animation system that generates realistic facial expressions and animations.

This project explores the possibilites of facial expression synthesis (FES) to generate synthetic DECA-Sequences with realistic facial expressions. Using NED (Neural Emotion Director) as a base the ECD allows the transformation of input data in real time. 

The ECD was created as part of a masters thesis at Hochschule Düsseldorf University of Applied Sciences

## Getting Started
Clone the repo:
  ```bash
  git clone https://github.com/m-117/ecd
  cd ecd
  ```

### Requirements  

Create a conda environment, using the provided ```environment.yml``` file.
```bash
conda env create -f environment.yml
```
Activate the environment.
```bash
conda activate ECD
```

Use the ```environment_render.yml``` file if you wish to render the created sequences as DECA shape images.

### Files
1. Follow the instructions in [DECA](https://github.com/YadiraF/DECA) (under the *Prepare data* section) to acquire the 3 files ('generic_model.pkl', 'deca_model.tar', 'FLAME_albedo_from_BFM.npz') and place them under "./DECA/data".
2. Download a model/checkpoint for text sentiment analysis that is compatible with the transformers library, for example this [checkpoint for DistilRoBERTa](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)

## Usage

To use the ECD you have to either prepare text data and DECA Sequences containing a fitting mouth movement or use NED to extract DECA Parameter from in-the-wild videos. 

The prepared input data can be transformed using the ```manipulate_expression()``` function in ```manipulator.py```

In order to render the manipulated sequences use the alternative conda environment and use ```create_inputs.py```

## Acknowledgements
I would like to thank the following great repositories that my code borrows from:
- [NED](https://github.com/foivospar/NED)
- [DECA](https://github.com/YadiraF/DECA)
- [MEAD](https://github.com/uniBruce/Mead)
- [Emotion English DistilRoBERTa-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)



