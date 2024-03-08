import sys
sys.path.append('./')

import json
from tqdm import tqdm
from utils.multiWoZUtils import MultiWoZSchema, getUpdataInPred
from configs.DiCoSConfig import DiCoSConfig
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

if __name__ == '__main__':
    config  = DiCoSConfig()
    config.load_from_file('./configs/DiCoSConfig.json')

    schema = MultiWoZSchema(config)

    with open(config.test_path, 'r', encoding='utf-8') as in_: # 读取数据集
        list_dialogue = json.load(in_)
    
    with open('./data/MultiWoZ/2.2/update_pred/cls_score_test_turn0.json', 'r', encoding='utf-8') as in_:
        pred_update = json.load(in_)


    list_y = []
    list_y_hat = []

    dict_gt_update = {}

    for dialogueIdx, dict_dialogue in tqdm(enumerate(list_dialogue), 'load dataset '):
        '''
            {
                "dialogue_idx": "MUL0012.json",
                "domains": [],
                "dialogue": []
            }
                
        '''
        last_state = schema.init_state() # 初始化状态字典
        for dialogueTurn in dict_dialogue['dialogue']: 
            '''
                {
                    "system_transcript": "",
                    "turn_idx": 0,
                    "belief_state": [],
                    "turn_label": [],
                    "transcript": "i need information on a hotel that include -s free parking please .",
                    "system_acts": [],
                    "domain": "hotel"
                }
            '''
            turnid = dict_dialogue['dialogue_idx'] + '_' + str(dialogueTurn['turn_idx'])
            current_state = schema.build_state(dialogueTurn['belief_state'])
            dict_update, dict_cate, dict_nocate = schema.getUpdateInfo(last_state, current_state)
            dict_update_pred = getUpdataInPred(config, schema, pred_update, turnid)

            dict_gt_update[turnid] = dict_update

            for slotName, value in dict_update_pred.items():
                
                list_y_hat.append(config.dict_update[value])
                list_y.append(config.dict_update[dict_update[slotName]])

            last_state = current_state

    with open('./data/MultiWoZ/2.2/update_pred/gt_zkh.json', 'w', encoding='utf-8') as out_:
        json.dump(dict_gt_update, out_, ensure_ascii=False, indent=4)

        
    print('准确'+str(accuracy_score(list_y, list_y_hat)))
    print("精确"+str(precision_score(list_y, list_y_hat, average=None)))
    print("recall"+str(recall_score(list_y, list_y_hat, average=None)))
    print("f1"+str(f1_score(list_y, list_y_hat, average=None)))