---
license: mit
language:
  - rna
  - dna
tags:
  - GFM
  - OmniGenome
widget:
- text: AGUCGGCAGAAAAGUUGGUGCUUAGACCACGCCC<mask>CUAGCCGCCGUAAUAAUAGAUAAAUAGGCG
---


# Bridging Sequence-Structure Alignment in RNA Foundation Models (AAAI 2025)

## Model Description

**OmniGenome** is an advanced RNA foundation model that introduces sequence-structure alignment to genomic modeling. The model bridges the gap between RNA sequences and their secondary structures, enabling bidirectional mappings that improve the flow of genomic information between RNA sequences and structures. With OmniGenome, researchers can achieve improved performance in RNA-related tasks, such as RNA design, secondary structure prediction, and various downstream genomic tasks. It also demon...

- **Model type**: Transformer-based (52M and 186M parameter versions)
- **Languages**: RNA sequences and structures
- **Pretraining**: The model is pretrained on RNA sequences from over 1,000 plant species from the OneKP database. Secondary structures were predicted using ViennaRNA.
- **Key Features**: 
  - Seq2Str (Sequence to Structure) and Str2Seq (Structure to Sequence) mapping
  - RNA design and secondary structure prediction
  - Generalizability to DNA genomic tasks

## Intended Use

This model is ideal for:
- RNA secondary structure prediction
- RNA design via structure-to-sequence mapping
- Genomic sequence understanding tasks, such as mRNA degradation rate prediction
- Transfer learning to DNA tasks, including promoter strength prediction, gene expression regression, and more

It is a valuable tool for researchers in RNA genomics, bioinformatics, and molecular biology.

## Limitations

OmniGenome is primarily trained on RNA data and its transferability to other genomic data (like human DNA) may require further finetuning. While it demonstrates excellent performance in in-silico experiments, in-vivo validation is yet to be performed.

## Training Data

OmniGenome was pretrained on large-scale RNA sequences from the OneKP initiative, which contains transcriptome data from 1,124 plant species. These sequences were processed and cleaned to ensure data quality, and secondary structures were annotated using ViennaRNA. The alignment between sequences and structures was a core part of the training process, enabling both Seq2Str and Str2Seq capabilities.

## Evaluation Results

OmniGenome was evaluated on multiple in-silico RNA benchmarks, including the EternaV2 RNA design benchmark, where it solved 74% of the puzzles, compared to only 3% by previous foundation models. It also achieved state-of-the-art performance in tasks such as mRNA degradation rate prediction and secondary structure prediction. In DNA-related tasks, OmniGenome achieved high F1 scores in tasks like chromatin accessibility prediction and polyadenylation site classification, even without any DNA-specific...

## How to Use

Here’s an example of how to load and use OmniGenome on Hugging Face:

``` python
from transformers import AutoTokenizer, AutoModel

# Load pre-trained model tokenizer
tokenizer = AutoTokenizer.from_pretrained("yangheng/OmniGenome")

# Load pre-trained model
model = AutoModel.from_pretrained("yangheng/OmniGenome")

# Example RNA sequence input
input_seq = "AUGGCUACUUUCG"

# Tokenize input
inputs = tokenizer(input_seq, return_tensors="pt")

# Perform inference
outputs = model(**inputs)
```

## Citation

If you use this model in your research, please cite the following:

Yang et al. OmniGenome: Bridging Sequence-Structure Alignment in RNA Foundation Models. [Link to paper]

## License

This model is released under the Apache 2.0 License.
