import networkx as nx
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class AliasSampling:

    # Reference: https://en.wikipedia.org/wiki/Alias_method

    #np.random.seed(42)

    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res

# to declare global hyperparameters

K = 2
alpha = 5
q = 0.3  # switch graph prob
dim = 64
T = 80  # num of new node pairs for each time aling networks

walk_len = 40  # random walk len
window = 10  # window size
walk_num = 30  # number of random walk started from one node

thresh = 0.5  # thresh for predict link

rand_seed = 0


class MakeIter(object):
    def __init__(self, generator_func, **kwargs):
        self.generator_func = generator_func
        self.kwargs = kwargs
    def __iter__(self):
        return self.generator_func(**self.kwargs)



# to generate random walk iter over the nodes of G1, G2 provided the alias sampler of each node, matching Pi

def walk_generator(G1, G2, Pi1, Pi2, n2word, idx2n, alias_Samplers, rand, walk_len=walk_len, walk_num=walk_num, q=q):
    
    n1, n2 = len(G1.nodes), len(G2.nodes)
    
    def random_walk(start_node, graph_idx):
        assert graph_idx in [1,2], "st_idx should indicate 1 or 2: G1 or G2"

        path = [n2word[(graph_idx, start_node)]]
        curr_node = start_node
        #print('Start-%s' %start_node, end='')
        while len(path) < walk_len: 
            # check if switch graph:
            if rand.uniform(0,1) < q: # stay in current graph
                
                if graph_idx == 1:
                    curr_neighbors = list(G1[curr_node])
                    curr_node = rand.choice(curr_neighbors)
                else:
                    curr_neighbors = list(G2[curr_node])
                    curr_node = rand.choice(curr_neighbors)
                                    
                path.append(n2word[(graph_idx, curr_node)])
                #print('st-%s-'%curr_node, end='')
                
            else: # switch graph
                
                if graph_idx == 1 and (curr_node in Pi1.keys()):
                    #print('Switched on matched node: ', end='')
                    #print('From %s '%n2word[(graph_idx, curr_node)], end=' ')
                    curr_node = Pi1[curr_node]
                    graph_idx = 2
                    #print('To %s '%n2word[(graph_idx, curr_node)])
                    
                elif graph_idx == 2 and (curr_node in Pi2.keys()):
                    #print('Switched on matched node: ', end='')
                    #print('From %s '%n2word[(graph_idx, curr_node)], end=' ')
                    curr_node = Pi2[curr_node]
                    graph_idx = 1
                    #print('To %s '%n2word[(graph_idx, curr_node)])
                
                else: 
                    sampler = alias_Samplers[(graph_idx, curr_node)] # sampler sample node idx, not node directly
                    
                    graph_idx = 1 if graph_idx == 2 else 2  # switch graph
                    
                    curr_node = idx2n[(graph_idx, sampler.sampling())]  # sampling in new graph, so need to update graph_idx before mapping id to node
                    
                path.append(n2word[(graph_idx, curr_node)])
                #print('SW-%s-' %curr_node, end='')
                
        #print()
        return path    
    
    nodes1 = [(1,u) for u in G1.nodes]
    nodes2 = [(2,v) for v in G2.nodes]
    nodes = nodes1 + nodes2
    
    for cnt in range(walk_num):
        rand.shuffle(nodes)
        for graph_idx, node in nodes:
            yield random_walk(start_node=node, graph_idx=graph_idx)  


# return degree_min, max of 0->k-hop neighbors of G, start from node u
def k_hop_min_max(G, u, k=K):
    k_hop_dict = nx.single_source_shortest_path_length(G, u, cutoff=k)
    degree_min = torch.full((k+1,), 0)
    degree_max = torch.full((k+1,), 0)
    
    for node in k_hop_dict.keys():
        d = G.degree(node)
        hop_idx = k_hop_dict[node]
        if (degree_min[hop_idx] == 0) or (degree_min[hop_idx] > d):
            degree_min[hop_idx] = d
        if (degree_max[hop_idx] == 0) or (degree_max[hop_idx] < d):
            degree_max[hop_idx] = d
    
    return degree_min, degree_max

# cross-graph embedding with random walk and skipgram

def cross_embed(G1, G2, Pi1, Pi2, dim=dim, seed=rand_seed, walk_len=walk_len, walk_num=walk_num, q=q, neg_sample=True):  
    np.random.seed(seed)
    rand = random.Random(seed)
    # Pi1, Pi2 is the node mappping. Pi2 = inverse Pi1
    # Pi1[u] = v
    # Pi2[v] = u
    
    # node dict of G1, G2
    n2word = {} # use when build random walk and retrieve embedding
    n2idx = {} # use when refer node index in cross_weight table
    idx2n = {}
    
    for i, node in enumerate(G1.nodes):
        n2word[(1,node)] = 'G1-%s'%i
        n2idx[(1,node)] = i
        idx2n[(1,i)] = node
    for i, node in enumerate(G2.nodes):
        n2word[(2,node)] = 'G2-%s'%i
        n2idx[(2,node)] = i
        idx2n[(2,i)] = node
    
    # cross-graph weight tensor w_u_v:
    n1, n2 = len(G1.nodes), len(G2.nodes)
    e1, e2 = len(G1.edges), len(G2.edges)
    rectified_factor = torch.sqrt(torch.tensor((e2/n2)/(e1/n1))).item()
    print('rectified_factor: ', rectified_factor)
    
    """
    cross_weight = torch.zeros((n1, n2))
    
    for u in G1.nodes:
        for v in G2.nodes:
            degree_min_u, degree_max_u = k_hop_min_max(G1, u, k=K)
            degree_min_u, degree_max_u = degree_min_u * rectified_factor + 1, degree_max_u * rectified_factor + 1
    
            degree_min_v, degree_max_v = k_hop_min_max(G2, v, k=K)
            degree_min_v, degree_max_v = degree_min_v + 1, degree_max_v + 1            
            
            f_u_v = torch.sum(torch.abs(torch.log(degree_min_u) - torch.log(degree_min_v)) + 
                              torch.abs(torch.log(degree_max_u) - torch.log(degree_max_v)))
            
            w_u_v = torch.exp(-alpha*f_u_v).item()
            
            cross_weight[n2idx[(1,u)], n2idx[(2,v)]] = w_u_v
    """        
            
            
    min_G1, max_G1 = torch.zeros((n1, K+1)), torch.zeros((n1, K+1))
    min_G2, max_G2 = torch.zeros((n2, K+1)), torch.zeros((n2, K+1))
    
    for u in G1.nodes:
        degree_min_u, degree_max_u = k_hop_min_max(G1, u, k=K)
        idx = n2idx[(1,u)]
        min_G1[idx] = degree_min_u
        max_G1[idx] = degree_max_u
    for v in G2.nodes:
        degree_min_v, degree_max_v = k_hop_min_max(G2, v, k=K)
        idx = n2idx[(2,v)]
        min_G2[idx] = degree_min_v
        max_G2[idx] = degree_max_v
    
    min_G1, max_G1 = min_G1 * rectified_factor + 1, max_G1 * rectified_factor + 1
    min_G2, max_G2 = min_G2 + 1, max_G2 + 1
    
    min_G1, max_G1 = min_G1.unsqueeze(1), max_G1.unsqueeze(1)
    min_G2, max_G2 = min_G2.unsqueeze(0), max_G2.unsqueeze(0)
    
    cross_weight = torch.abs(torch.log(min_G1) - torch.log(min_G2)) + torch.abs(torch.log(max_G1) - torch.log(max_G2))
    print('cross_weight shape: ', cross_weight.shape)
    cross_weight = torch.exp(-alpha*torch.sum(cross_weight, dim=2))
    print('cross_weight shape: ', cross_weight.shape)
            
    
    print('Finished building cross_weight table')
    
    # build Alias sampler for each node of G1, G2:
    alias_Samplers = {}  # dict of alias samplers for nodes
    
    for u in G1.nodes:
        #print('u: %s' %u)
        #print('n2idx[(1,u)]: %s'%n2idx[(1,u)])
        prob_u = cross_weight[n2idx[(1,u)]] / torch.sum(cross_weight[n2idx[(1,u)]])
        #print("length prob_u: %d" %len(prob_u))
        alias_Samplers[(1,u)] = AliasSampling(prob_u)
    for v in G2.nodes:
        #print('v: %s' %v)
        #print('n2idx[(2,v)]: %s'%n2idx[(2,v)])
        prob_v = cross_weight[:, n2idx[(2,v)]] / torch.sum(cross_weight[:, n2idx[(2,v)]])
        #print("length prob_v: %d" %len(prob_v))
        alias_Samplers[(2,v)] = AliasSampling(prob_v)
    
    print('Finished building Alias Samplers')
        
    # TODO: define RandomWalk
    
    walk_iter = MakeIter(walk_generator, G1=G1, G2=G2, Pi1=Pi1, Pi2=Pi2, n2word=n2word, idx2n=idx2n, alias_Samplers=alias_Samplers, rand=rand, walk_len=walk_len, walk_num=walk_num, q=q)
    
    #walk_gen = walk_generator(G1=G1, G2=G2, Pi1=Pi1, Pi2=Pi2, n2word=n2word, idx2n=idx2n, alias_Samplers=alias_Samplers, rand=rand, walk_len=walk_len, walk_num=walk_num, q=q)
    #for walk in walk_gen:
    #    print('len walk: %d' %len(walk))
    #    print(walk)

    
    print('Embedding Compound graph...')
    if neg_sample:
        model = Word2Vec(walk_iter, size=dim, window=window, min_count=0, sg=1, hs=0, negative=5)
    else:
        model = Word2Vec(walk_iter, size=dim, window=window, min_count=0, sg=1, hs=1)

    print('Done embedding')
    
    return model.wv, n2word, n2idx, idx2n
    




# Network Alignment

def node_alignment(G1, G2, Pi1, Pi2, N1, N2, embed_dict, n2word, n2idx, idx2n, T=T):
    # embed_dict is the model.wv: Word2VecKeyedVectors in gensim Word2Vec model
    
    n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
    
    #N1 = set()
    #N2 = set()
    
    def sim_emb(u, v):
        # it is necessary to specify order u, v: u is in G1, v is in G2
        word_u = n2word[(1,u)]
        word_v = n2word[(2,v)]
        sim = embed_dict.similarity(word_u, word_v)
        
        if sim < 0:
            sim = 0
        return sim
        
    def sim_jc(u, v):
        assert len(Pi1) > 0 and len(Pi2) > 0, "Do not use Jaccard when S1, S2 are empty"
            
        neighbors_u = set(G1[u]).intersection(set(Pi1.keys()))
        pi_neighbors_u = set([Pi1[node] for node in neighbors_u])
        
        neighbors_v = set(G2[v]).intersection(set(Pi2.keys()))
        
        intersec = pi_neighbors_u.intersection(neighbors_v)
        union = pi_neighbors_u.union(neighbors_v)
        
        sim = len(intersec)/len(union)
        
        return sim
    
    def sim_attr(u, v):
        assert False, "Need to implement sim_attr"
    
    
    def create_neighbor_set(graph_idx, Pi):
        S = Pi.keys()
        assert len(S) > 0, "Pi is empty. Need to match first pair"
        neighbors_set = set()
        G = G1 if graph_idx == 1 else G2
        for node in S:
            neighbors_set.update(set(G[node]))
        
        neighbors = neighbors_set.difference(S)
        
        return neighbors
    
    def update_neighbor_set(graph_idx, Pi, N):
        assert len(N) > 0, "N is empty! use create_neighbor_set for initialization"
        S = Pi.keys()
        G = G1 if graph_idx == 1 else G2
        just_matched_nodes = N.intersection(S)
        N = N.difference(S)
        for node in just_matched_nodes:
            N.update(set(G[node]).difference(S))  # neighbors of node that not in S -> add to N
        
        return N
    
    
    if len(Pi1) == n1 or len(Pi2) == n2:
        print('Already matched every possible node pairs')
        return Pi1, Pi2, N1, N2
    
    elif len(Pi1) == 0:
        assert len(Pi2) == 0, "len Pi1, Pi2 must be equal"
        
        similarities = np.zeros((n1, n2))
        #all_words2 = [n2word[(2,v)] for v in G2.nodes]   #NEED TO CHECK IF THE ORDER IS CORRECT
        all_words2 = np.empty(n2, dtype=object)
        for v in G2.nodes:
            all_words2[n2idx[(2,v)]] = n2word[(2,v)]
        for u in G1.nodes:
            vector_u = embed_dict[n2word[(1,u)]]
            vectors_all = embed_dict[all_words2]
            similarities[n2idx[(1,u)]] = embed_dict.cosine_similarities(vector_u, vectors_all)
            
        similarities = np.where(similarities < 0, 0, similarities) # filter sim(u,v) = max(cosine(u,v), 0)
        #print('shape of similarities: ', similarities.shape)
        
        for _ in range(T):
            if len(Pi1) = n1 or len(Pi2) = n2:   # if matched all nodes -> break
                break
            i, j = np.unravel_index(similarities.argmax(), similarities.shape)
            similarities[i], similarities[:,j] = -1, -1
            Pi1[idx2n[(1,i)]] = idx2n[(2,j)]
            Pi2[idx2n[(2,j)]] = idx2n[(1,i)]
        
        print([(n2word[(1,u)],n2word[(2,Pi1[u])]) for u in Pi1.keys()])
        print(Pi2)
        
    else:
        if len(N1) == 0:
            assert len(N2) == 0, "len N1, N2 at the beggining must equal 0"
            N1, N2 = create_neighbor_set(1, Pi1), create_neighbor_set(2, Pi2)
        else:
            N1, N2 = update_neighbor_set(1, Pi1, N1), update_neighbor_set(2, Pi2, N2)
            
        print('Size of N1: %s, N2: %s' %(len(N1), len(N2)))
        
        similarities = np.zeros((len(N1), len(N2)))
        n2i = {}
        i2n = {}
        for i, u in enumerate(N1):
            n2i[(1,u)] = i
            i2n[(1,i)] = u
            for j, v in enumerate(N2):
                n2i[(2,v)] = j
                i2n[(2,j)] = v
                sim = sim_emb(u,v) * sim_jc(u,v)
                similarities[i,j] = sim
        
        for _ in range(T):
            if len(Pi1) = n1 or len(Pi2) = n2:  # if matched all nodes -> break
                break
            i, j = np.unravel_index(similarities.argmax(), similarities.shape)
            similarities[i], similarities[:,j] = -1, -1
            Pi1[i2n[(1,i)]] = i2n[(2,j)]
            Pi2[i2n[(2,j)]] = i2n[(1,i)]
        
        print([(n2word[(1,u)],n2word[(2,Pi1[u])]) for u in Pi1.keys()])
        print(Pi2)
    
    return Pi1, Pi2, N1, N2



