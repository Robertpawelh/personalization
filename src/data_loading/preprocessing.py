import pandas as pd
import numpy as np
import re

def filter_invalid_values(dset):
    dset = dset.loc[dset['type'] != 'Uszkodzony']
    dset = dset.loc[dset['brand'] == 'iPhone']
    dset = dset.dropna(subset='price')

    return dset

def get_specification(name, description):
    pattern = re.compile(r'\b(iphone 11 pro max|iphone 11 pro|iphone 11)\b.*?(64|128|256|512)', re.IGNORECASE)
    match = re.search(pattern, name + " " + description)

    if match:
        model = match.group(1).lower()
        storage = int(match.group(2).lower().split('gb')[0].strip())

        if model == 'iphone 11':
            model = 'base'
        elif model == 'iphone 11 pro':
            model = 'pro'
        elif model == 'iphone 11 pro max':
            model = 'max'

        return f'{model}_{storage}'
    else:
        return np.nan

def drop_specs_with_less_than_n_days(dset, min_days):
    specs = dset['specification'].unique()
    specs_to_drop = []
    for spec in specs:
        spec_dset = dset[dset['specification'] == spec]
        if (spec_dset['scrap_time'].max() - spec_dset['scrap_time'].min()).days < min_days:
            specs_to_drop.append(spec)
    return dset.loc[~dset['specification'].isin(specs_to_drop)]

def preprocessing(dset, model_params):
    dset['scrap_time'] = pd.to_datetime(dset['scrap_time'])
    dset = filter_invalid_values(dset)
    dset['specification'] = dset.apply(lambda x: get_specification(x['name'], x['description']), axis=1)
    dset.dropna(subset=['specification'], inplace=True)
    dset = drop_specs_with_less_than_n_days(dset, model_params.n_days_in + model_params.n_days_out)

    return dset
