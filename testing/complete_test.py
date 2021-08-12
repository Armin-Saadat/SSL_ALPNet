import os
import json
import warnings
import pandas as pd


def read_complete_test(root_tests_path='/HDD/SSL_ALPNet_models/', model_name='NO_NAME'):
    """
    Args:
        root_tests_path: root of test logs
        model_name: name of the model you are testing
    Does:
        writes a csv file of mean_dice and mean_prec for all snapshots
    """
    if model_name == 'NO_NAME':
        warnings.warn("You didn't specified model_name!!")

    tests_path = root_tests_path + "exps/myexperiments_MIDDLE_0/mySSL_test_vfold_CHAOST2_Superpix_sets_0_1shot"
    all_metrics = []
    all_mean_dice = []
    all_mean_prec = []
    for folder in sorted(os.listdir(tests_path)):
        if folder[0] == '.':
            continue
        try:
            a = int(folder)
        except ValueError:
            raise Exception('SOME NON INTEGER FOLDER IN TESTING DIR')

        f_path = os.path.join(tests_path, folder)
        metrics = json.load(open(os.path.join(f_path, 'metrics.json')))
        print(f'folder {folder} metrics loaded')
        all_metrics.append(metrics)

        # some other metrics could be exported too
        all_mean_dice.append(round(metrics.get('mar_val_batches_meanDice').get('values')[0], 4))
        all_mean_prec.append(round(metrics.get('mar_val_batches_meanPrec').get('values')[0], 4))

    os.mkdir(os.path.join(root_tests_path, model_name))
    df = pd.DataFrame()
    df['snapshots'] = [i * 5000 for i in range(len(all_metrics))]
    df['mean_dice'] = all_mean_dice
    df['mean_prec'] = all_mean_prec
    os.system(f'touch {model_name}.csv')
    df.to_csv(os.path.join(root_tests_path, model_name, f'{model_name}.csv'))


read_complete_test(root_tests_path="/Users/kian/Desktop/tests", model_name='NO_NAME')
