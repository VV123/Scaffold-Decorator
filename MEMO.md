
# MEMO

## Notes

- Extracting index w/ variable length using gather ([Link](https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4))
   ```
   idx_in_x1 = x2_idx.unsqueeze(-1) # B, sub_seq_len (64), 1
   idx_in_x1 = idx_in_x1.repeat(1, 1, self.d_model)
   x2 = torch.gather(output, 1, idx_in_x1)
   ```
   
- Hugging face transformer ([Link](https://github.com/pytorch/tutorials/issues/719))
