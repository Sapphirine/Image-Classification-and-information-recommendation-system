import os,sys
from decoder import *
import os
import dynet as dynet
import random
import matplotlib.pyplot as plt
import numpy as np


cwd = os.getcwd()
print cwd

actionFile = open(cwd+"/voca/vocabs.actions", 'r').read().strip().split('\n')
action_dict={}
r_action_dict={}
for i in actionFile:
    action_dict[i.split(" ")[0]]= float(i.split(" ")[1])
    r_action_dict[float(i.split(" ")[1])]= i.split(" ")[0]
    
labelFile = open(cwd+"/voca/vocabs.labels", 'r').read().strip().split('\n')
label_dict={}  
r_label_dict={}
for i in labelFile:
    label_dict[i.split(" ")[0]]= float(i.split(" ")[1])
    r_label_dict[float(i.split(" ")[1])]= i.split(" ")[0]
   
    
posFile = open(cwd+"/voca/vocabs.pos", 'r').read().strip().split('\n')
pos_dict={}   
r_pos_dict={}
for i in posFile:
    pos_dict[i.split(" ")[0]]= float(i.split(" ")[1])
    r_pos_dict[float(i.split(" ")[1])]= i.split(" ")[0]

wordFile = open(cwd+"/voca/vocabs.word", 'r').read().strip().split('\n')
word_dict={}   
r_word_dict={}
for i in wordFile:
    word_dict[i.split(" ")[0]] = float(i.split(" ")[1])
    r_word_dict[float(i.split(" ")[1])]= i.split(" ")[0]

    
def actionid2action_str(id):
    return r_action_dict[id]

def action2id(action):
    return action_dict[action]

def labelid2label_str(id):
    return r_label_dict[id]

def label2id(label):
    return label_dict[label] if label in label_dict else label_dict['<null>']

def posid2pos_str(id):
    return r_pos_dict[id]

def pos2id(pos):
    return pos_dict[pos] if pos in pos_dict else pos_dict['<null>']

def wordid2word_str(id):
    return r_word_dict[id]

def word2id(word):
    return word_dict[word] if word in word_dict else word_dict['<null>']

def num_actions():
    return len(r_action_dict)

def num_labels():
    return len(r_label_dict)

def num_poses():
    return len(r_pos_dict)

def num_words():
    return len(word_dict)

model = dynet.Model()


# In[11]:


# assign the algorithm for backpropagation updates.
updater = dynet.AdamTrainer(model)


# In[12]:


# create embeddings for words and tag features.
#word_embed_dim, pos_embed_dim = 100, 50
word_embed_dim, pos_embed_dim, d_embed_dim = 64,32,32
word_embedding = model.add_lookup_parameters((num_words(), word_embed_dim))
pos_embedding = model.add_lookup_parameters((num_poses(), pos_embed_dim))
d_embedding = model.add_lookup_parameters((num_labels(), d_embed_dim))
#word_embedding = model.add_lookup_parameters((num_words(), word_embed_dim))
#tag_embedding = model.add_lookup_parameters((num_tag_feats(), pos_embed_dim))


# In[13]:


# assign transfer function
transfer = dynet.rectify  # can be dynet.logistic or dynet.tanh as well.


# In[14]:


# define the input dimension for the embedding layer.
# here we assume to see two words after and before and current word (meaning 5 word embeddings)
# and to see the last two predicted tags (meaning two tag embeddings)
#input_dim = 5 * word_embed_dim + 2 * pos_embed_dim
input_dim = 20 * word_embed_dim + 20* pos_embed_dim + 12 * d_embed_dim

#hidden_dim, minibatch_size = 200, 1000
hidden_dim1, hidden_dim2, minibatch_size = 200,200,1000
#hidden_dim3 = 400

# define the hidden layer.
#hidden_layer = model.add_parameters((hidden_dim, input_dim))
hidden_layer1 = model.add_parameters((hidden_dim1, input_dim))


# define the hidden layer bias term and initialize it as constant 0.2.
#hidden_layer_bias = model.add_parameters(hidden_dim, init=dynet.ConstInitializer(0.2))
hidden_layer_bias1 = model.add_parameters(hidden_dim1, init=dynet.ConstInitializer(0.2))

hidden_layer2 = model.add_parameters((hidden_dim2, hidden_dim1))
hidden_layer_bias2 = model.add_parameters(hidden_dim2, init=dynet.ConstInitializer(0.2))

#hidden_layer3 = model.add_parameters((hidden_dim3, hidden_dim2))
#hidden_layer_bias3 = model.add_parameters(hidden_dim3, init=dynet.ConstInitializer(0.2))

# define the output weight.
output_layer = model.add_parameters((num_actions(), hidden_dim2))
#output_layer = model.add_parameters((num_actions(), hidden_dim3))
# define the bias vector and initialize it as zero.
output_bias = model.add_parameters(num_actions(), init=dynet.ConstInitializer(0))


# # Implementing the Forward function
# 

# In[15]:


def forward(features):
   # extract word and tags ids
   #word_ids = [word2id(word_feat) for word_feat in features[0:5]]
   #tag_ids = [feat_tag2id(tag_feat) for tag_feat in features[5:]]
   word_ids= [word2id(word_feat) for word_feat in features[0:20]]
   pos_ids= [pos2id(pos_feat) for pos_feat in features[20:40]]
   label_ids=[label2id(label_feat) for label_feat in features[40:]]

    
   # extract word embeddings and tag embeddings from features
   #word_embeds = [word_embedding[wid] for wid in word_ids]
   #tag_embeds = [tag_embedding[tid] for tid in tag_ids]
   word_embeds = [word_embedding[wid] for wid in word_ids]
   pos_embeds = [pos_embedding[pid] for pid in pos_ids]
   d_embeds = [d_embedding[lid] for lid in label_ids]

   # concatenating all features (recall that '+' for lists is equivalent to appending two lists)
   #embedding_layer = dynet.concatenate(word_embeds + tag_embeds)
   embedding_layer = dynet.concatenate(word_embeds + pos_embeds + d_embeds)
   # calculating the hidden layer
   # .expr() converts a parameter to a matrix expression in dynetnet (its a dynetnet-specific syntax).
   #hidden = transfer(hidden_layer.expr() * embedding_layer + hidden_layer_bias.expr())
   hidden1 = transfer(hidden_layer1.expr() * embedding_layer + hidden_layer_bias1.expr())
   hidden2 = transfer(hidden_layer2.expr() * hidden1 + hidden_layer_bias2.expr() )
   #hidden3 = transfer(hidden_layer3.expr() * hidden2 + hidden_layer_bias3.expr() )

   # calculating the output layer
   #output = output_layer.expr() * hidden + output_bias.expr()
   output = output_layer.expr() * hidden2 + output_bias.expr()
   #output = output_layer.expr() * hidden3 + output_bias.expr()
   # return a list of outputs
   return output


loss_values = []
plt.ion()
ax = plt.gca()
ax.set_xlim([0, 10])
ax.set_ylim([0, 3])
plt.title("Loss over time")
plt.xlabel("Minibatch")
plt.ylabel("Loss")
plt.show()


# In[17]:


def plot(loss_values):
    print loss_values
    ax.set_xlim([0, len(loss_values)+10])
    ax.plot(loss_values)
    plt.draw()
    try:
       plt.pause(0.0001)
    except: pass

train_data = open(cwd+"/data/train.data", 'r').read().strip().split('\n')


# In[19]:


def train_iter(train_data):
        losses = [] # minibatch loss vector
        random.shuffle(train_data) # shuffle the training data.

        for line in train_data:
            fields = line.strip().split(' ')
            #features, label, gold_label = fields[:-1], fields[-1], tag2id(fields[-1])
            features, label, gold_label = fields[0:52], fields[52], action2id(fields[52])
            result = forward(features)

            # getting loss with respect to negative log softmax function and the gold label; and appending to the minibatch losses.
            loss = dynet.pickneglogsoftmax(result, gold_label)
            losses.append(loss)

            if len(losses) >= minibatch_size:
                minibatch_loss = dynet.esum(losses) / len(losses) # now we have enough loss values to get loss for minibatch
                minibatch_loss.forward() # calling dynetnet to run forward computation for all minibatch items
                minibatch_loss_value = minibatch_loss.value() # getting float value of the loss for current minibatch

                # printing info and plotting
                loss_values.append(minibatch_loss_value)
                
                if len(loss_values)%10==0: plot(loss_values)
                #print loss_values
                minibatch_loss.backward() # calling dynetnet to run backpropagation
                updater.update() # calling dynet to change parameter values with respect to current backpropagation

                # empty the loss vector and refresh the memory of dynetnet
                losses = []
                dynet.renew_cg()

        dynet.renew_cg()

for i in range(7):
    print 'epoch', (i+1)
    train_iter(train_data)
    dynet.renew_cg()
print 'finished training!'





class DepModel:
    def __init__(self):
        '''
            You can add more arguments for examples actions and model paths.
            You need to load your model here.
            actions: provides indices for actions.
            it has the same order as the data/vocabs.actions file.
        '''
        # if you prefer to have your own index for actions, change this.
        self.actions = ['SHIFT', 'LEFT-ARC:rroot', 'LEFT-ARC:cc', 'LEFT-ARC:number', 'LEFT-ARC:ccomp', 'LEFT-ARC:possessive', 'LEFT-ARC:prt', 'LEFT-ARC:num', 'LEFT-ARC:nsubjpass', 'LEFT-ARC:csubj', 'LEFT-ARC:conj', 'LEFT-ARC:dobj', 'LEFT-ARC:nn', 'LEFT-ARC:neg', 'LEFT-ARC:discourse', 'LEFT-ARC:mark', 'LEFT-ARC:auxpass', 'LEFT-ARC:infmod', 'LEFT-ARC:mwe', 'LEFT-ARC:advcl', 'LEFT-ARC:aux', 'LEFT-ARC:prep', 'LEFT-ARC:parataxis', 'LEFT-ARC:nsubj', 'LEFT-ARC:<null>', 'LEFT-ARC:rcmod', 'LEFT-ARC:advmod', 'LEFT-ARC:punct', 'LEFT-ARC:quantmod', 'LEFT-ARC:tmod', 'LEFT-ARC:acomp', 'LEFT-ARC:pcomp', 'LEFT-ARC:poss', 'LEFT-ARC:npadvmod', 'LEFT-ARC:xcomp', 'LEFT-ARC:cop', 'LEFT-ARC:partmod', 'LEFT-ARC:dep', 'LEFT-ARC:appos', 'LEFT-ARC:det', 'LEFT-ARC:amod', 'LEFT-ARC:pobj', 'LEFT-ARC:iobj', 'LEFT-ARC:expl', 'LEFT-ARC:predet', 'LEFT-ARC:preconj', 'LEFT-ARC:root', 'RIGHT-ARC:rroot', 'RIGHT-ARC:cc', 'RIGHT-ARC:number', 'RIGHT-ARC:ccomp', 'RIGHT-ARC:possessive', 'RIGHT-ARC:prt', 'RIGHT-ARC:num', 'RIGHT-ARC:nsubjpass', 'RIGHT-ARC:csubj', 'RIGHT-ARC:conj', 'RIGHT-ARC:dobj', 'RIGHT-ARC:nn', 'RIGHT-ARC:neg', 'RIGHT-ARC:discourse', 'RIGHT-ARC:mark', 'RIGHT-ARC:auxpass', 'RIGHT-ARC:infmod', 'RIGHT-ARC:mwe', 'RIGHT-ARC:advcl', 'RIGHT-ARC:aux', 'RIGHT-ARC:prep', 'RIGHT-ARC:parataxis', 'RIGHT-ARC:nsubj', 'RIGHT-ARC:<null>', 'RIGHT-ARC:rcmod', 'RIGHT-ARC:advmod', 'RIGHT-ARC:punct', 'RIGHT-ARC:quantmod', 'RIGHT-ARC:tmod', 'RIGHT-ARC:acomp', 'RIGHT-ARC:pcomp', 'RIGHT-ARC:poss', 'RIGHT-ARC:npadvmod', 'RIGHT-ARC:xcomp', 'RIGHT-ARC:cop', 'RIGHT-ARC:partmod', 'RIGHT-ARC:dep', 'RIGHT-ARC:appos', 'RIGHT-ARC:det', 'RIGHT-ARC:amod', 'RIGHT-ARC:pobj', 'RIGHT-ARC:iobj', 'RIGHT-ARC:expl', 'RIGHT-ARC:predet', 'RIGHT-ARC:preconj', 'RIGHT-ARC:root']
        # write your code here for additional parameters.
        # feel free to add more arguments to the initializer.

    def score(self, str_features):
        '''
        :param str_features: String features
        20 first: words, next 20: pos, next 12: dependency labels.
        DO NOT ADD ANY ARGUMENTS TO THIS FUNCTION.
        :return: list of scores
        '''
        # change this part of the code.
        #f = sentence.strip().split(' ').strip().rsplit(' ',1)[0]
        #action = decode(f)
        #output = [f + '\t' + action for f, action in zip(f, action)]
        #writer.write('\n'.join(output) + '\n\n')
        #writer.close()
        output = forward(str_features)

       # getting list value of the output
        scores = output.npvalue()
        #print scores
       # getting best tag
        #best_action_id = np.argmax(scores)

       # assigning the best tag
        #ts.append(tagid2tag_str(best_tag_id))
    
        
        return scores
    

    

if __name__=='__main__':
    m = DepModel()
    input_p = os.path.abspath(sys.argv[1])
    output_p = os.path.abspath(sys.argv[2])
    Decoder(m.score, m.actions).parse(input_p, output_p)
