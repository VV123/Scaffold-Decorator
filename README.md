# Scaffold Decorator
This project works on AI-based drug discovery using a generative model to generate novel drug-like compounds. 
The major challenge of generative model-based drug discovery is the unlimited sample space with millions of novel compounds to be generated. 
We propose to bound the generative drug-like structures to a reasonable space by modifying the scaffold (partial molecules with explicit attachment points) of existing drugs. 

Here is our conceptual design:

<img width="706" alt="Screen Shot 2023-10-21 at 4 55 20 PM" src="https://github.com/VV123/Scaffold-Decorator/assets/9030237/19b064ee-a4ee-431c-bab8-323d2541c7a5">



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
