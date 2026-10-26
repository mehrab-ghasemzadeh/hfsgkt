from collections import defaultdict
import torch

class EpisodeSampler:
    """Index-only sampling: deterministic per episode, no image decoding to index classes."""
    def __init__(self,dataset,n_way=5,n_support=1,n_query=15,seed=42):
        if min(n_way,n_support,n_query)<1:
            raise ValueError('n_way, n_support, n_query must be positive')
        self.dataset,self.n_way,self.n_support,self.n_query,self.seed = dataset,n_way,n_support,n_query,seed
        self.class_to_indices = defaultdict(list)
        for i,label in enumerate(dataset.targets):
            self.class_to_indices[label].append(i)
        self.eligible = sorted(c for c,ids in self.class_to_indices.items() if len(ids)>=n_support+n_query)
        if len(self.eligible)<n_way:
            raise ValueError(f'Only {len(self.eligible)} eligible classes; need {n_way}, each with >= {n_support+n_query} images')

    def sample_indices(self,episode_id):
        generator = torch.Generator().manual_seed(self.seed+int(episode_id))
        chosen = torch.randperm(len(self.eligible),generator=generator)[:self.n_way].tolist()
        selected = [self.eligible[position] for position in chosen]
        records = getattr(self.dataset,'records',None)
        def group(index):
            if records is None:
                return f'image:{index}'
            value = records[index].get('group','').strip()
            return f'group:{value}' if value else f'image:{index}'
        # First choose all support images. Exclude their patients from all queries,
        # even across pathology classes. Retry when a support draw exhausts a class.
        for attempt in range(100):
            result = dict(support=[],support_y=[],query=[],query_y=[],classes=[])
            for label,cls in enumerate(selected):
                candidates = self.class_to_indices[cls]
                ids = torch.randperm(len(candidates),generator=generator)[:self.n_support].tolist()
                result['support'].extend(candidates[j] for j in ids)
                result['support_y'].extend([label]*self.n_support)
                result['classes'].append(self.dataset.classes[cls])
            support_groups = {group(index) for index in result['support']}
            for label,cls in enumerate(selected):
                candidates = [i for i in self.class_to_indices[cls] if group(i) not in support_groups]
                if len(candidates)<self.n_query:
                    break
                ids = torch.randperm(len(candidates),generator=generator)[:self.n_query].tolist()
                result['query'].extend(candidates[j] for j in ids)
                result['query_y'].extend([label]*self.n_query)
            else:
                return result
        raise ValueError('Unable to draw patient-disjoint support/query after 100 attempts; inspect patient groups and episode sizes')

    def sample(self,episode_id):
        record = self.sample_indices(episode_id)
        return (torch.stack([self.dataset[i][0] for i in record['support']]),torch.tensor(record['support_y']),
                torch.stack([self.dataset[i][0] for i in record['query']]),torch.tensor(record['query_y']))
