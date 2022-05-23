import json
import torch
import ibm_model.ibm_em as em

def load():
    image_data = torch.load("../datasets/flickr8k/resnet_features.pt")
    images = dict(zip(image_data['filenames'], image_data['features']))
    meta = json.load(open("../datasets/flickr8k/dataset.json"))
    for image in meta['images']:
        if image['split'] == 'train':
            for sentence in image['sentences']:
                yield (images[image['filename']].tolist(), sentence['tokens'])
            
def load_data():
    src, tgt = zip(*load())
    return (src, tgt)


def train():
    src, tgt = load_data()
    model = em.M1(src, tgt, online=True, weighted=True)
    model.iterate()
    tt = model.translation_table()
    emb = torch.tensor([ [ v.get(i, 0) for i in range(2048) ] for v in tt.values() ]).float()
    return emb

def pairwise_cosine(input_a, input_b):
   normalized_input_a = torch.nn.functional.normalize(input_a)  
   normalized_input_b = torch.nn.functional.normalize(input_b)
   return torch.mm(normalized_input_a, normalized_input_b.T)

def nn(vocab, emb):
    sim = pairwise_cosine(emb, emb)
    top5 = {}
    for i, word in enumerate(vocab):
    
        js = sim[i].argsort()[-5:-1]
        top5[word] = [vocab[j] for j in js]
    return top5

        
        
        

