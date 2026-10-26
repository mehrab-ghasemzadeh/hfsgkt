# Engineering and mathematical decisions

This is a new hybrid experimental method, not an exact reproduction of either base paper. The following decisions must be stated in the eventual Method/Implementation sections.

| Issue in supplied guide or difference from base papers | Implemented behavior |
|---|---|
| Hybrid fusion indexed the batch dimension as a layer dimension | Broadcast `[B,L,1]` weights against `[B,L,d]` transformed features |
| Unnormalized fused mid-level vector despite a unit-norm requirement | Normalize each transformed layer and the fused vector |
| Guide says softmax layer weights; HFDA Eq.4 prints a normalized linear score | Use stable positive softmax weights as the chosen hybrid design |
| Standard ResNet18 84px feature sizes were incorrect | Unmodified torchvision stem produces block2/3/4 spatial sizes 11/6/3 |
| Query images were averaged across different classes | Independent graph and prediction per query; no averaging and no query-query graph edges |
| k=8 failed for 5-way 1-shot | Clamp k to N-1, exclude diagonal before topk and add self loops |
| Guide called union-symmetrized kNN mutual kNN | Implement union kNN and name it correctly |
| Guide promised attention coefficients but used a separate unconstrained tau MLP | Read the final GAT attention from receiver query to sender supports, average heads, exclude query self-loop, renormalize supports |
| Original KT uses optimized potentially signed coefficients | Hybrid tau is nonnegative attention, sums to one; it is not the KT sparse solver |
| Guide hidden width 512 per head would produce 2048 concatenated dimensions | hidden_dim means TOTAL width across all heads; default total 512, 128 per head |
| Graph smoothing can erase original node information | Residual projection plus LayerNorm at every GAT layer |
| Pixel masking was applied to features in the guide | Apply independent masks to support images before encoding, shared over RGB channels |
| Guide included an untrained class embedding head | Train GNN head with labeled TRAIN-query CE; never with test-query labels |
| Different source/target classes reused the source class IDs | Source cosine classifier and separate target heads with independent class mappings |
| Four adaptive terms in HFDA were simplified to three and log regularizers omitted | Four losses: source FSL, target FSL, DA, DS; minimize sum(phi*L)-0.5*sum(log(phi)), exact simplex via softmax |
| A softmax constraint made lambda*(sum(phi)-1) identically zero | Explicit constrained parameterization of HFDA Eq.10-11; no claim to implement independent primal-dual ascent Eq.12 |
| Uniform-target domain classifier can learn a constant output by itself | Alternate true-domain CE training of DA discriminator on detached features with KL-uniform feature updates under a fixed discriminator; this is an engineering extension |
| Stage1a high-only CE never trained mid selectors | Supervise both normalized mid and high paths using the source classifier |
| HFDA trains end to end; guide splits/freeze training | Keep the requested three-stage hybrid with frozen backbone in Stage1b, frozen encoder/BN in Stage2 |
| Stage2 in KT adapts to novel task; guide trains shared episodic modules | Learn a shared query-conditioned episodic head on the allowed train domain; no per-test-episode gradient fitting in this implementation |
| HFDA(+) uses extra augmentation/label propagation | Neither HFDA(+) label propagation nor its reported accuracy is claimed here |
| Raw-logit mixing used incompatible score scales | Mix probabilities, expose alpha and residual temperature, return log probabilities |
| Scalar MSE in example averaged over features, unlike KT squared norm | Sum squared reconstruction error across feature dimensions, average over queries |
| Positive entropy penalty can encourage collapse when minimized | Optional entropy term is SUBTRACTED from loss to encourage spread; default zero |
| Freeze flags alone let BatchNorm drift | Keep encoder/backbone in eval as well as disabling gradients |
| Iterator reset repeatedly sampled first batch | Step-seeded random mini-batches with reproducible indices |
| Evaluation used population std | Report normal-approximation episode CI with sample std (ddof=1); at least two episodes |

Stage2 loss is `mse_weight*MSE + consistency_weight*CS + ce_weight*(vote_NLL + GNN_CE) - entropy_weight*H(tau)`.

Query images used for graph inference are unlabeled inputs. Training query labels in CE come only from the configured training split/domain. Test query labels are read only after inference to compute metrics. Graph conditioning on an unlabeled query must be described separately from fitting parameters to test labels, which is not done.

GAT is a dense PyTorch implementation of standard additive attention for small episode graphs. It requires no PyG extension wheels. Its head concatenation, self-loop masking and attention normalization follow the usual GAT definition; see [GATConv mathematical reference](https://pytorch-geometric.readthedocs.io/en/2.4.0/generated/torch_geometric.nn.conv.GATConv.html). Dense per-query graphs scale quadratically in the support count, so large-shot settings require memory-aware chunk sizes.

Checkpoints contain state dictionaries and primitive/tensor RNG state, loaded with `weights_only=True` and explicit stage checks. Atomic writes use a temporary file and `os.replace`. Config inheritance is a small YAML deep-merge implementation with dot-key overrides, not Hydra.

Known limits: no automatic verification that external manifests match published splits; no perceptual duplicate detector; no DDP/AMP; no per-episode optimization during test; no original KT/HFDA baseline reimplementation; no claim of multi-seed real-data performance. Reproduce those baselines using their official implementations and matched data access before publishing comparison claims.
