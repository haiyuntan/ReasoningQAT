# Legacy numerical correction

Legacy analysis had 1638/1691 low-entropy positions with zero teacher tail under float32 subtraction. Those tail-ratio conclusions are provisional and superseded by this run. Model logits remain BF16; FP64 statistics use excluded-top1 logsumexp and log1p normalization. Primary log ratio has no epsilon; epsilon and FP32 sensitivity remain secondary diagnostics. No old arrays or conclusions are silently overwritten. Original layer14 plots remain valid representation artifacts.
