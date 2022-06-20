
import argparse
import sys
from data_loader import *
from LSTMSentAvg import LSTMSentAvg
from LSTMParSeq import LSTMParSeq
from LSTMSemRel import LSTMSemRel
from CNNPosTag import CNNPosTag
from FusionSemSyn import FusionSemSyn
from train_neural_models import *
import pickle

sys.path.insert(0, os.getcwd())

dirname, filename = os.path.split(os.path.abspath(__file__))
root_dir = "/".join(dirname.split("/")[:-1])

run_dir = os.path.join(root_dir, "runs")

parser = argparse.ArgumentParser()

parser.add_argument("--task", type=str, default="class")

# model params
parser.add_argument("--model_type", type=str,
                    default="sent_avg")  # clique, doc_seq
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--dropout", type=float, default=0)
parser.add_argument("--lstm_dim", type=int, default=100)
parser.add_argument("--hidden_dim", type=int, default=200,
                    help="hidden layer dimension")
parser.add_argument("--l2_reg", type=float, default=0)

# training
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=10)
# for debugging with subset of data
parser.add_argument("--train_data_limit", type=int, default=-1)
parser.add_argument("--lr_decay", type=str, default="none")

# vectors
parser.add_argument("--vector_type", default="glove",
                    help="specify vector type glove/word2vec/none")
parser.add_argument("--glove_path", type=str,
                    default="data/GloVe/glove.840B.300d.txt")
parser.add_argument("--embedding_dim", type=int,
                    default=300, help="vector dimension")
parser.add_argument("--case_sensitive", action="store_true",
                    help="activate this flag if vectors are case-sensitive (don't lower-case the data)")

# per-experiment settings
parser.add_argument("--model_name", type=str)
parser.add_argument("--data_dir", default="data/",
                    help="path to the data directory")
parser.add_argument("--train_corpus", type=str)
parser.add_argument("--test_corpus", type=str)
parser.add_argument("--cross_val", type=int,
                    default=0, help="Use the cross validation setting")
parser.add_argument("--pos_tag", type=int,
                    default=0, help="Use the pos tag setting")
parser.add_argument("--tag_filter", type=int,
                    default=0, help="POS Tag filtering is used")

args = parser.parse_args()
if args.model_name is None:
    print("Specify name of experiment")
    sys.exit(0)
if args.train_corpus is None:
    print("Specify train corpus")
    sys.exit(0)
if args.test_corpus is None:
    args.test_corpus = args.train_corpus

params = {
    'top_dir': root_dir,
    'run_dir': run_dir,
    'model_name': args.model_name,
    'data_dir': args.data_dir,
    'train_corpus': args.train_corpus,
    'test_corpus': args.test_corpus,
    'task': args.task,
    'train_data_limit': args.train_data_limit,
    'lr_decay': args.lr_decay,
    'model_type': args.model_type,
    'glove_file': args.glove_path,
    'vector_type': args.vector_type,
    'embedding_dim': args.embedding_dim,  # word embedding dim
    'case_sensitive': args.case_sensitive,
    'learning_rate': args.learning_rate,
    'dropout': args.dropout,  # 1 = no dropout, 0.5 = dropout
    'hidden_dim': args.hidden_dim,
    'lstm_dim': args.lstm_dim,
    'l2_reg': args.l2_reg,
    'batch_size': args.batch_size,
    'num_epochs': args.num_epochs,
    'cross_val' : args.cross_val,
    'pos_tag' : args.pos_tag,
    'tag_filter' : args.tag_filter
    
}

if not os.path.exists(params['run_dir']):
    os.mkdir(params['run_dir'])
model_dir = os.path.join(params['run_dir'], params["model_name"])
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
params['model_dir'] = model_dir

# save parameters
with open(os.path.join(model_dir, params['model_name'] + '.params'), 'w', encoding='utf-8') as param_file:
    for key, parameter in params.items():
        param_file.write("{}: {}".format(key, parameter) + "\n")
        print((key, parameter))

start = time.time()
if params['vector_type'] == 'glove':
    params['vector_path'] = params['glove_file']

# load data
data = Data(params)
vectors = None

# Utiliser GloVe sérialisé  

embeddings = pickle.load(open('../api/pickle_files/word_embeds.pkl', 'rb'))
word_to_idx = pickle.load(open('../api/pickle_files/word_to_idx.pkl', 'rb'))
idx_to_word = pickle.load(open('../api/pickle_files/idx_to_word.pkl', 'rb'))
data.word_embeds = embeddings
data.word_to_idx = word_to_idx
data.idx_to_word = idx_to_word

# if params['vector_type'] != 'none':
#     vectors, vector_dim = data.load_vectors()
#     params['embedding_dim'] = vector_dim

# dev_docs = None
if params['vector_type'] == 'none':  # init random vectors
    vectors = data.rand_vectors(len(data.word_to_idx))

# Modèle de classification de la fusion entre le niveau sémantique et syntaxique avec validation croisée
if params['cross_val'] == 1 and params['task'] == 'class' and params['model_type'] == 'fusion_sem_syn' and params['pos_tag'] == 1 :
    data_docs_cnn = data.read_data_class_cv_tag(params)
    data_docs_sem = data.read_data_class_cv(params)
    model_fusion = FusionSemSyn(params, data_cnn=data_docs_cnn, data_semrel=data_docs_sem, data_obj=data)
    train_fusion(params, data_docs_cnn, data_docs_sem, data, model_fusion)

# Modèle de classification du niveau syntaxique avec la validation croisée
elif params['cross_val'] == 1 and params['task'] == 'class' and params['model_type'] == 'cnn_pos_tag' and params['pos_tag'] == 1 :
    data_docs = data.read_data_class_cv_tag(params)
    model = CNNPosTag(params, data) 
    train_cv(params, data_docs, data, model)

# Modèle de classification du niveau syntaxique
elif params['cross_val'] == 0 and params['pos_tag'] == 1 and params['task'] == 'class' and params['model_type'] == 'cnn_pos_tag':
    training_docs = data.read_data_class_tag(params, 'train')
    test_docs = data.read_data_class_tag(params, 'test')
    model = CNNPosTag(params, data)
    train(params, training_docs, test_docs, data, model)
    
# Modèle de régression du niveau sémantique pour capturer la transition entre les paragraphes
elif params['cross_val'] == 0 and params['pos_tag'] == 1 and params['task'] == 'score_pred' and params['model_type'] == 'cnn_pos_tag':
    training_docs = data.read_data_class_tag(params, 'train')
    test_docs = data.read_data_class_tag(params, 'test')
    model = CNNPosTag(params, data)
    train(params, training_docs, test_docs, data, model)

# Modèle de classification du niveau sémantique pour capturer la transition entre les phrases seulement avec la validation croisée
elif params['cross_val'] == 1 and params['task'] == 'class' and params['model_type'] == 'sent_avg':
    data_docs = data.read_data_class_cv(params)
    model = LSTMSentAvg(params, data)
    train_cv(params, data_docs, data, model)
    
# Modèle de classification du niveau sémantique pour capturer la transition entre les phrases seulement
elif params['cross_val'] == 0 and params['task'] == 'class' and params['model_type'] == 'sent_avg':
    training_docs = data.read_data_class(params, 'train')
    test_docs = data.read_data_class(params, 'test')
    model = LSTMSentAvg(params, data)
    best_test_acc = train(params, training_docs, test_docs, data, model)

# Modèle de régression du niveau sémantique pour capturer la transition entre les phrases seulement
elif params['cross_val'] == 0 and params['task'] == 'score_pred' and params['model_type'] == 'sent_avg':
    training_docs = data.read_data_class(params, 'train')
    test_docs = data.read_data_class(params, 'test')
    model = LSTMSentAvg(params, data)
    best_test_acc = train(params, training_docs, test_docs, data, model)
    
# Modèle de classification du niveau sémantique pour capturer la transition entre les paragraphes avec la validation croisée
elif params['cross_val'] == 1 and params['task'] == 'class' and params['model_type'] == 'par_seq':
    data_docs = data.read_data_class_cv(params)
    model = LSTMParSeq(params, data)
    train_cv(params, data_docs, data, model)
    
# Modèle de classification du niveau sémantique pour capturer la transition entre les paragraphes
elif params['cross_val'] == 0 and params['task'] == 'class' and params['model_type'] == 'par_seq':
    training_docs = data.read_data_class(params, 'train')
    test_docs = data.read_data_class(params, 'test')
    model = LSTMParSeq(params, data)
    best_test_acc = train(params, training_docs, test_docs, data, model)
    
# Modèle de régression du niveau sémantique pour capturer la transition entre les paragraphes
elif params['cross_val'] == 0 and params['task'] == 'score_pred' and params['model_type'] == 'par_seq':
    training_docs = data.read_data_class(params, 'train')
    test_docs = data.read_data_class(params, 'test')
    model = LSTMParSeq(params, data)
    best_test_acc = train(params, training_docs, test_docs, data, model)

# Modèle de classification de la fusion du niveau sémantique entre les phrases et les paragraphes avec la validation croisée
elif params['model_type'] == 'sem_rel' and params['cross_val'] == 1 and params['task'] == 'class':
    data_docs = data.read_data_class_cv(params)
    model = LSTMSemRel(params, data)
    train_cv(params, data_docs, data, model)

# Modèle de classification de la fusion du niveau sémantique entre les phrases et les paragraphes
elif params['model_type'] == 'sem_rel' and params['cross_val'] == 0 and params['task'] == 'class':
    training_docs = data.read_data_class(params, 'train')
    test_docs = data.read_data_class(params, 'test')
    model = LSTMSemRel(params, data)
    train(params, training_docs, test_docs, data, model)