params = {}

params['csv'] = './data_list/train.csv'
params['batchsize'] = 64
params['num_works'] = 8
params['gpu'] = [0, 1]
params['pretrain'] = False
params['lr'] = 1e-4
params['weight_decay'] = 1e-5
params['max_epoch'] = 100
params['save_path'] = './model/model.pth'

