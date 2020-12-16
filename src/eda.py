import os
import pandas as pd
from .drive_api_download import download_data

class BCW_Explorer:
    '''
    Class to explore the dataset
    '''
    def __init__(self, nrows=None):
        '''
        '''
        def get_data() -> bool:
            os.system("mkdir -p ./data")
            return download_data("1SOIlZxCtx2VRmVj7MycqCqpncJU6NgmSQ6kq4Ie20fk", "data/wbcd.csv")
            
        def load_data() -> None:
            try:
                data = pd.read_csv(
                    'data/wbcd.csv',
                    header=0,
                    # names=pd.np.array(header),
                    nrows=nrows
                )

            except Exception as e:
                print('Error while loading data: {0}'.format(e))

            # drops unnecessary id column
            data = data.drop("id", axis=1)

            # transform labels into binary
            labels = data["diagnosis"].map({'M': 1, 'B': 0}).astype('int')

            # enforcing multi-level index
            summaries = ['mean', 'std', 'meanmax3']
            multilevel =\
                pd.concat(
                    [
                        data.iloc[:, cols]
                        .rename(
                            columns={
                                x: attribute
                                for x, attribute
                                in zip(
                                    cols,
                                    [
                                        'radius', 'texture',
                                        'perimeter', 'area',
                                        'smoothness', 'compactness',
                                        'concavity_intensity',
                                        'concavity_count',
                                        'symmetry', 'fractal_dimension'
                                    ]
                                )
                            }
                        )
                        for cols in pd.np.array_split(
                            range(1, len(data.columns)), len(summaries)
                        )
                    ],
                    axis=1,
                    keys=summaries,
                    names=['summaries', 'attributes']
                )

            multilevel['label'] = labels
            return multilevel

        if get_data():
            data = load_data()
            self.holdout = data.sample(frac=.2)
            self.data = data[~data.index.isin(self.holdout.index)]
        else:
            raise Exception("Could not download dataset")
