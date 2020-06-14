#feature_network_config.py
Config = {}
normal_config = {
'fold_index':2,
"initial_lr": 0.001,
#Initial learning rate
"mode" : "train",#"train" or "test"
"model_name": "feature coding network",
"batch_size": 16,
"num_workers": 8,
#Multithreaded reading data
"epoch_num": 400,
"classes":['Chacha','Foxtrot','Jive','Pasodoble','Quickstep','Rumba','Salsa','Samba',
'Slowwaltz','Tango','Viennesewaltz','Waltz','Wcswing'],
#the music genres need to be predicted
"checkpoint": "/home/zwj/mgr/checkpoint",
"path_train": "/data/zwj/feature/cqt/train/cqt_train.pkl", 
#Training data path
'''
for example:
'/data/zwj/feature/transfer/train/transfer_train.pkl'
'/data/zwj/feature/hpss/train/harmonic_train.pkl'
'/data/zwj/feature/hpss/train/percussive_train.pkl'
'/data/zwj/feature/melspectrogram/train/melspectrogram_train.pkl'
'/data/zwj/feature/cqt/train/cqt_train.pkl'
'''
"path_test": "/data/zwj/feature/cqt/test/cqt_test.pkl",
#testing data path
'''
for example:
'/data/zwj/feature/transfer/test/transfer_test.pkl'
'/data/zwj/feature/hpss/test/harmonic_test.pkl'
'/data/zwj/feature/hpss/test/percussive_test.pkl'
'/data/zwj/feature/melspectrogram/test/melspectrogram_test.pkl'
'/data/zwj/feature/cqt/test/cqt_test.pkl'
'''
 }
feature = {
"model_best_feature": "/data/zwj//checkpoint/model_best_feature.pth.tar",
"checkpoint_feature": "/data/zwj/checkpoint/checkpoint_feature.pth.tar",
"train_feature": "/data/zwj/checkpoint/train_feature.txt",
"save_result" : "/data/zwj/fearture_e/{}/output_result.pkl",
"extract_feature": "/data/zwj/fearture_e/{}/extract_feature.pkl"
}

mel_6_14 = {
"mel_model_best": "/home/zwj/mgr/checkpoint/mel_model_best_{}.pth.tar",
"mel_checkpoint": "/home/zwj/mgr/checkpoint/mel_checkpoint_{}.pth.tar",
"mel_train_process_txt": "/home/zwj/mgr/train_process_txt/mel_train_process_{}.txt",
"save_result" : "/home/zwj/mgr/extract_coding/mel/{}/mel_output_result_{}.pkl",
"extract_feature": "/home/zwj/mgr/extract_coding/mel/{}/mel_extract_feature_{}.pkl"
}

cqt_6_14 = {
"cqt_model_best": "/home/zwj/mgr/checkpoint/cqt_model_best_{}.pth.tar",
"cqt_checkpoint": "/home/zwj/mgr/checkpoint/cqt_checkpoint_{}.pth.tar",
"cqt_train_process_txt": "/home/zwj/mgr/train_process_txt/cqt_train_process_{}.txt",
"save_result" : "/home/zwj/mgr/extract_coding/cqt/{}/cqt_output_result_{}.pkl",
"extract_feature": "/home/zwj/mgr/extract_coding/cqt/{}/cqt_extract_feature_{}.pkl"
}
percussive_6_14 = {
"percussive_model_best": "/home/zwj/mgr/checkpoint/percussive_model_best_{}.pth.tar",
"percussive_checkpoint": "/home/zwj/mgr/checkpoint/percussive_checkpoint_{}.pth.tar",
"percussive_train_process_txt": "/home/zwj/mgr/train_process_txt/percussive_train_process_{}.txt",
"save_result" : "/home/zwj/mgr/extract_coding/percussive/{}/percussive_output_result_{}.pkl",
"extract_feature": "/home/zwj/mgr/extract_coding/percussive/{}/percussive_extract_feature_{}.pkl"
}
harmonic_6_14 = {
"harmonic_model_best": "/home/zwj/mgr/checkpoint/harmonic_model_best_{}.pth.tar",
"harmonic_checkpoint": "/home/zwj/mgr/checkpoint/harmonic_checkpoint_{}.pth.tar",
"harmonic_train_process_txt": "/home/zwj/mgr/train_process_txt/harmonic_train_process_{}.txt",
"save_result" : "/home/zwj/mgr/extract_coding/harmonic/{}/harmonic_output_result_{}.pkl",
"extract_feature": "/home/zwj/mgr/extract_coding/harmonic/{}/harmonic_extract_feature_{}.pkl"
}
scatter_6_14 = {
"scatter_model_best": "/home/zwj/mgr/checkpoint/scatter_model_best_{}.pth.tar",
"scatter_checkpoint": "/home/zwj/mgr/checkpoint/scatter_checkpoint_{}.pth.tar",
"scatter_train_process_txt": "/home/zwj/mgr/train_process_txt/scatter_train_process_{}.txt",
"save_result" : "/home/zwj/mgr/extract_coding/scatter/{}/scatter_output_result_{}.pkl",
"extract_feature": "/home/zwj/mgr/extract_coding/scatter/{}/scatter_extract_feature_{}.pkl"
}
transfer_6_14 = {
"transfer_model_best": "/home/zwj/mgr/checkpoint/transfer_model_best_{}.pth.tar",
"transfer_checkpoint": "/home/zwj/mgr/checkpoint/transfer_checkpoint_{}.pth.tar",
"transfer_train_process_txt": "/home/zwj/mgr/train_process_txt/transfer_train_process_{}.txt",
"save_result" : "/home/zwj/mgr/extract_coding/transfer/{}/transfer_output_result_{}.pkl",
"extract_feature": "/home/zwj/mgr/extract_coding/transfer/{}/transfer_extract_feature_{}.pkl"
}

timbre_6_14 = {
"timbre_model_best": "/home/zwj/mgr/checkpoint/timbre_model_best_{}.pth.tar",
"timbre_checkpoint": "/home/zwj/mgr/checkpoint/timbre_checkpoint_{}.pth.tar",
"timbre_train_process_txt": "/home/zwj/mgr/train_process_txt/timbre_train_process_{}.txt",
"save_result" : "/home/zwj/mgr/extract_coding/transfer/{}/timbre_output_result_{}.pkl",
"extract_feature": "/home/zwj/mgr/extract_coding/transfer/{}/timbre_extract_feature_{}.pkl"
}

st_6_14 = {
"st_model_best": "/home/zwj/mgr/checkpoint/st_model_best_{}.pth.tar",
"st_checkpoint": "/home/zwj/mgr/checkpoint/st_checkpoint_{}.pth.tar",
"st_train_process_txt": "/home/zwj/mgr/train_process_txt/st_train_process_{}.txt",
"save_result" : "/home/zwj/mgr/extract_coding/st/{}/st_output_result_{}.pkl",
"extract_feature": "/home/zwj/mgr/extract_coding/st/{}/st_extract_feature_{}.pkl"
}
mst_6_14 = {
"mst_model_best": "/home/zwj/mgr/checkpoint/mst_model_best_{}.pth.tar",
"mst_checkpoint": "/home/zwj/mgr/checkpoint/mst_checkpoint_{}.pth.tar",
"mst_train_process_txt": "/home/zwj/mgr/train_process_txt/mst_train_process_{}.txt",
"save_result" : "/home/zwj/mgr/extract_coding/mst/{}/mst_output_result_{}.pkl",
"extract_feature": "/home/zwj/mgr/extract_coding/mst/{}/mst_extract_feature_{}.pkl"
}

Config["mel_6_14"] = mel_6_14
Config["mst_6_14"] = mst_6_14

Config["st_6_14"] = st_6_14
Config["scatter_6_14"] = scatter_6_14
Config["cqt_6_14"] = cqt_6_14
Config["mel_6_14"] = mel_6_14
Config["harmonic_6_14"] = harmonic_6_14
Config["percussive_6_14"] = percussive_6_14
Config["transfer_6_14"] = transfer_6_14
Config["feature"]= feature
Config["normal_config"] = normal_config
