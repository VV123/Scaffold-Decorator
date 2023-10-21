# Scaffold Decorator
This project works on AI-based drug discovery using a generative model to generate novel drug-like compounds. 
The major challenge of generative model-based drug discovery is the unlimited sample space with millions of novel compounds to be generated. 
We propose to bound the generative drug-like structures to a reasonable space by modifying the scaffold (partial molecules with explicit attachment points) of existing drugs. 

## Dataset
- https://registry.opendata.aws/chembl/
## How to run

```
python3 main.py --batch_size 512 --layer 3 \
                --d_model 512 --path model.pt \
                --mode [train, infer] --epoch 500
```

## Funding

[AWS Cloud Credit for Research](https://aws.amazon.com/government-education/research-and-technical-computing/cloud-credit-for-research/) 
