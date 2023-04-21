import pickle as pkl

def read_data(filename):
    """Reads the .txt file and filters out the target sequences to return them in a list

    Args:
        filename (str): path to file 

    Returns:
        str[]: list of all target sequences in the data
    """
    file = open(filename, 'r')
    targets = []
    for line in file:
        line_split = line.split()
        targets.append(line_split[0])

    return targets

def read_data_with_features(filename):
    """Reads the .txt file returns them in a list

    Args:
        filename (str): path to file 

    Returns:
        str[]: list of all target sequences in the data
    """
    file = open(filename, 'r')
    data_points = []
    for line in file:
        line_split = line.split()
        line_split = [line_split[0]] + [float(i) for i in line_split[1:]]
        data_points.append(line_split)

    return data_points

def read_pkl_raw_data(filename):
    """Reads raw pkl data and returns list of target sequences

    Args:
        filename (str): path to file
    
    Returns:
        str[]: list of all target sequences in the data
    """
    replicate_raw_data = pkl.load(open(filename,'rb'))
    sequences = []
    long_sequences = []
    for item in replicate_raw_data:
        sequences.append(item[6])
        long_sequences.append(item[1])

    return long_sequences, sequences

def observed_to_expected_CpG_filter_1(targets, exp_obs_ratio, GC_criterium):
    """Filters CG dense target sequences based on the criteria taken from: https://en.wikipedia.org/wiki/CpG_site#CpG_islands

    Args:
        targets (str[]): list of all target sequences
        ratio (float): ratio observed / expected CpG content on which to filter

    Returns:
        str[]: list of all CG dense target sequences
    """
    filtered_targets = []
    for target in targets:
        C_occurrences = target.count('C')
        G_occurrences = target.count('G')
        observed = target.count('CG')
        GC_content = (observed * 2) / len(target)
        expected = (C_occurrences * G_occurrences) / len(target)
        if observed > 2 and ((observed / expected) > exp_obs_ratio) and (((C_occurrences + G_occurrences) / len(target)) > GC_criterium):
            filtered_targets.append([target, observed / expected, (C_occurrences + G_occurrences) / len(target)])
    
    return filtered_targets

def observed_to_expected_CpG_filter_2(targets, exp_obs_ratio, GC_criterium):
    """Filters CG dense target sequences based on the criteria from 2nd paper on taken from: https://en.wikipedia.org/wiki/CpG_site#CpG_islands

    Note: this criterium is not used in the final analysis but was used to see if a signficant difference was observed between these two
    different criteria. Conclusion: no signficant difference in sequneces filtered. They filter almost the exact same sequences.
    Args:
        targets (str[]): list of all target sequences
        ratio (float): ratio observed / expected CpG content on which to filter

    Returns:
        str[]: list of all CG dense target sequences
    """
    filtered_targets = []
    for target in targets:
        C_occurrences = target.count('C')
        G_occurrences = target.count('G')
        observed = target.count('CG')
        expected = (((C_occurrences + G_occurrences)/2)**2) / len(target)
        if ((observed / expected) > exp_obs_ratio) and (((C_occurrences + G_occurrences) / len(target)) > GC_criterium):
            filtered_targets.append([target, observed / expected, (C_occurrences + G_occurrences) / len(target)])

    return filtered_targets