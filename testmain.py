import random
import os
from tcramodel import *
from copy import deepcopy
from dataLoad import init_dataset
from transformerMTI.predicttcra import *
# from transformerMTI.test import *
from roc import  *
from torch.utils.data import DataLoader

if __name__ == "__main__":


    # TODO 使用os模块获取路径，方便迁移
    # 获取上级路径
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))

    cuda= torch.cuda.is_available()
    # 显卡id
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    devices_ids = [0,1]

    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    DATASET2 = "genemiRNA_train"
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""


    ##数据加载器的参数
    Testdata_address = '../data/test.csv'


    gene_dim = 100
    rna_dim = 34
    hid_dim = 64
    n_layers = 4
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    batch = 32
    lr = 1e-4
    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 1
    # iteration = 300
    SPPSize = 64
    feaSize = 64
    filterNum = 128
    contextSizeList = [1, 3, 5]
    in_planes = 64
    out_planes = 64

    encoder = Encoder(gene_dim, rna_dim, hid_dim, dropout, device,SPPSize, feaSize, filterNum, contextSizeList)
    decoder = Decoder(rna_dim, hid_dim, n_layers, n_heads, pf_dim, MultiHeadAttentionWithACmix, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)

    testSet = init_dataset(Testdata_address)
    my_test_data = DataLoader(testSet, batch_size=1, shuffle=True)


    m = torch.load("output/model/deepmirnatar")
    from collections import OrderedDict

    new_dict = OrderedDict()
    ll = list(m.keys())
    for k in ll:
        val = m[k]
        m[k[7:]] = val
            # new_dict[key] = model.state_dict()[key]
        del m[k]
    model.load_state_dict(m)

    # TODO: gpu支持
    if cuda:
        model.cuda()
            # 开启并行模式
        model = torch.nn.DataParallel(model).cuda()
    tester = Tester(model)

     # """Output files."""
    file_AUCs = os.path.join(BASE_DIR, 'transformerMTI/output/test/test.txt')
    AUCs = ('acc_test\tpre_test\tf1_test\tAUC_test\tPRC_test\tsen_test\tspe_test\tmcc_test')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

        # TODO： 使用迭代器
    my_test_data = deepcopy(my_test_data)

    acc_test, pre_test, rec_test, f1_test, AUC_test, PRC_test, sen_test, spe_test, mcc_test = tester.test(
            my_test_data)

    AUCs = [round(acc_test, 7), round(pre_test, 7), round(f1_test, 7), round(AUC_test, 7), round(PRC_test, 7),
                round(sen_test, 7), round(spe_test, 7), round(mcc_test, 7)]
    tester.save_AUCs(AUCs, file_AUCs)
    print('\t'.join(map(str, AUCs)))



