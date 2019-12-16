import sys
import pandas as pd
sys.path.append('/home/ottok92/Dev/elogroup2/')


class BCW_Explorer:
    '''
    Class to explore the dataset
    '''
    def __init__(self, nrows=None):
        '''
        '''
        def load_data() -> None:
            try:
                data = pd.read_csv(
                    '../data/raw/wdbc.data',
                    header=None,
                    # names=pd.np.array(header),
                    nrows=nrows
                )

            except Exception as e:
                print('Error while loading data: {0}'.format(e))

            # drops unnecessary id column
            data = data.drop(0, axis=1)

            # transform labels into binary
            labels = data[1].map({'M': 1, 'B': 0}).astype('int')

            # enforcing multi-level index
            summaries = ['mean', 'std', 'meanmax3']
            multilevel =\
                pd.concat(
                    [
                        data[cols]
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
                            range(2, 32), len(summaries)
                        )
                    ],
                    axis=1,
                    keys=summaries,
                    names=['summaries', 'attributes']
                )

            multilevel['label'] = labels
            return multilevel

        data = load_data()
        self.holdout = data.sample(frac=.2)
        self.data = data[~data.index.isin(self.holdout.index)]
