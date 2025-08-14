import numpy as np
import pandas as pd

def linearInterpolation(discharge_data):
    """
    Interpolates voltage, temperature, and discharge capacity data for each battery cycle
    into a 30x30 representation.

    Parameters:
    - discharge_data:  battery discharge dataset with columns "BatteryIndex", "CycleIndex", "V", "T", "Qd"
   
    Returns:
    - V_interpol: 30x30 voltage matrix.
    - T_interpol:  30x30 temperature matrix.
    - Qd_interpol: 30x30 discharge capacity matrix.
    """

    voltage_range = np.linspace(3.6, 2.0, 900)

    V_interpol = {}
    T_interpol = {}
    Qd_interpol = {}

    # Get unique battery index and cycle index pairs
    battery_cycles = discharge_data[['BatteryIndex', 'CycleIndex']].drop_duplicates()
    # for each row in battery cycles, get the battery and cycle index
    for _, row in battery_cycles.iterrows():
        battery_index = row['BatteryIndex']
        cycle_index = row['CycleIndex']
        
        # Get the data of that battery and cycle pair where there voltage is neg (the discharge data)
        cycle_data = discharge_data[
        (discharge_data['BatteryIndex'] == battery_index) & 
        (discharge_data['CycleIndex'] == cycle_index)
        ].copy()

        # Sort by voltage (numpy inter needs it sorted)
        cycle_data = cycle_data.sort_values('V', ascending=False)

        # Now we will create interpolation functions

        # Remove duplicate voltage values
        unique_indices = ~cycle_data['V'].duplicated()
        voltage_unique = cycle_data['V'][unique_indices].values
        temp_unique = cycle_data['T'][unique_indices].values
        qd_unique = cycle_data['Qd'][unique_indices].values

        sorted_data = pd.DataFrame({'V': voltage_unique, 'T': temp_unique, 'Qd': qd_unique}).sort_values('V')
        voltage_sorted = sorted_data['V'].values
        temp_sorted = sorted_data['T'].values
        qd_sorted = sorted_data['Qd'].values
            
        voltage_range_sorted = voltage_range[::-1]  # make ascending

        v_interp = np.interp(voltage_range_sorted, voltage_sorted, voltage_sorted)
        t_interp = np.interp(voltage_range_sorted, voltage_sorted, temp_sorted)
        qd_interp = np.interp(voltage_range_sorted, voltage_sorted, qd_sorted)
        
        # Flip back to original order (3.6 to 2V)
        v_interp = v_interp[::-1]
        t_interp = t_interp[::-1]
        qd_interp = qd_interp[::-1]
        
        # Reshape to 30x30 arrays
        key = (battery_index, cycle_index)
        V_interpol[key] = v_interp.reshape(30, 30)
        T_interpol[key] = t_interp.reshape(30, 30)
        Qd_interpol[key] = qd_interp.reshape(30, 30)

    return V_interpol, T_interpol, Qd_interpol


def hreshapeData(V_interpol, T_interpol, Qd_interpol):
    """
    Combines voltage, temperature, and discharge capacity matrices into 3 channel arrays
    and computes normalized Remaining Useful Life labels.

    Parameters:
    - V_interpol: 30x30 voltage matrix.
    - T_interpol: 30x30 temperature matrix.
    - Qd_interpol: 30x30 discharge capacity matrix.

    Returns:
    - train_data: Combined 3 channel matrices (V, T, Qd).
    - train_rul_data: Normalized RUL values, where RUL = (max cycle - current cycle) / 2000.
    """

    # Get all the battery cycle combinations from the voltage dictionary
    all_keys = list(V_interpol.keys())
    num_samples = len(all_keys)
    #print(f"Processing {num_samples} battery cycle combinations")

    # Initializing arrays
    train_data = np.zeros((num_samples, 30, 30, 3))
    train_rul_data = np.zeros(num_samples)

    # Go through each battery/cycle combination
    for i, key in enumerate(all_keys):
        battery_index, cycle_index = key
        
        # Extract the 30x30 matrices for the current battery cycle combination
        voltage_matrix = V_interpol[key]      # 30x30 voltage data
        temp_matrix = T_interpol[key]         # 30x30 temperature data
        qd_matrix = Qd_interpol[key]          # 30x30 discharge capacity data
        
        # This is like the 3 channeled RGB
        train_data[i, :, :, 0] = voltage_matrix
        train_data[i, :, :, 1] = temp_matrix
        train_data[i, :, :, 2] = qd_matrix

        # RUL would be total cycles for this battery - the current cycle
        # Find all cycles and then the max which is the final cycle (the num of cycles the battery did)
        battery_cycles = [c for (b, c) in all_keys if b == battery_index]
        max_cycle_for_battery = max(battery_cycles)

        rul = max_cycle_for_battery - cycle_index
        train_rul_data[i] = rul / 2000.0  # normalized by maximum expected life
    
    return train_data, train_rul_data

def compute_global_min_max(data):
    """
    Computes global minimum and maximum values for normalization per channel.

    Parameters:
    - data: data with shape (N, C, H, W). N samples, C channels, HxW dimensions.

    Returns:
    - mins: Minimum values for each channel.
    - maxs: Maximum values for each channel.
    """
    C = data.shape[1]
    mins = []
    maxs = []
    for ch in range(C):
        mins.append(np.min(data[:, ch, :, :]))
        maxs.append(np.max(data[:, ch, :, :]))
    return np.array(mins), np.array(maxs)

def normalize_with_global_min_max(data, mins, maxs):
    """
    Normalizes each channel of the dataset to the [0, 1] range using min/max values.

    Parameters:
    - data: Data to normalize. (Shape (N, C, H, W))
    - mins: Minimum values for each channel.
    - maxs: Maximum values for each channel.

    Returns:
    - normalized_data: Data scaled to [0, 1] per channel. (Shape (N, C, H, W))
    """
    data = np.copy(data)
    for ch in range(data.shape[1]):
        data[:, ch, :, :] = (data[:, ch, :, :] - mins[ch]) / (maxs[ch] - mins[ch] + 1e-8)
    return data



def prepare_battery_data_for_cnn(V_interpol, T_interpol, Qd_interpol, total_batteries=40):
    """
    Splits battery data into training, validation, and test sets, reshapes them, and prepares corresponding RUL labels for CNN input.

    Parameters:
    - V_interpol: 30x30 voltage matrix.
    - T_interpol: 30x30 temperature matrix.
    - Qd_interpol: 30x30 discharge capacity matrix.
    - total_batteries: Total number of batteries in the dataset (default is 40 for mathworks data).

    Returns:
    - train_data: Training set.
    - train_rul_data: Normalized RUL labels for training set.
    - val_data: Validation set.
    - val_rul_data: Normalized RUL labels for validation set.
    - test_data: Test set. Shape (N_test, 30, 30, 3)
    - test_rul_data: Normalized RUL labels for test set.
    """
    
    # Get battery indicies for train, validation and test split (copying which indicies from mathworks)
    # Doing 1 and 0 because python index starts at 0, whereas matlab starts at 1
    test_battery_indices = list(range(1, total_batteries + 1, 8)) 
    val_battery_indices = list(range(0, total_batteries + 1, 8))
    
    all_battery_indices = list(range(total_batteries))
    excluded_indices = set(test_battery_indices + val_battery_indices)
    train_battery_indices = [idx for idx in all_battery_indices if idx not in excluded_indices]
    
    # Making sure we got the right indicies
    #print(f"Train batteries (count: {len(train_battery_indices)}): {train_battery_indices}")
    #print(f"Validation batteries (count: {len(val_battery_indices)}): {val_battery_indices}")  
    #print(f"Test batteries (count: {len(test_battery_indices)}): {test_battery_indices}")
    
    # Filter dictionaries for each dataset to only include the specific battery indicies
    def filter_dict_by_battery_indices(data_dict, battery_indices):
        filtered_dict = {}
        for (batt_idx, cycle_idx), data in data_dict.items():
            if batt_idx in battery_indices:
                filtered_dict[(batt_idx, cycle_idx)] = data
        return filtered_dict
    
    # Create filtered dictionaries for training data
    print("Filtering training data")
    train_V = filter_dict_by_battery_indices(V_interpol, train_battery_indices)
    train_T = filter_dict_by_battery_indices(T_interpol, train_battery_indices)
    train_Qd = filter_dict_by_battery_indices(Qd_interpol, train_battery_indices)
    
    # Create filtered dictionaries for validation data
    print("Filtering validation data")
    val_V = filter_dict_by_battery_indices(V_interpol, val_battery_indices)
    val_T = filter_dict_by_battery_indices(T_interpol, val_battery_indices)
    val_Qd = filter_dict_by_battery_indices(Qd_interpol, val_battery_indices)
    
    # Create filtered dictionaries for test data
    print("Filtering test data")
    test_V = filter_dict_by_battery_indices(V_interpol, test_battery_indices)
    test_T = filter_dict_by_battery_indices(T_interpol, test_battery_indices)
    test_Qd = filter_dict_by_battery_indices(Qd_interpol, test_battery_indices)
    
    # Reshape data for each split
    print("reshaping training data")
    train_data, train_rul_data = hreshapeData(train_V, train_T, train_Qd)
    
    print("reshaping validation data") 
    val_data, val_rul_data = hreshapeData(val_V, val_T, val_Qd)
    
    print("reshaping test data")
    test_data, test_rul_data = hreshapeData(test_V, test_T, test_Qd)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Training RUL data shape: {train_rul_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Validation RUL data shape: {val_rul_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test RUL data shape: {test_rul_data.shape}")
    
    return train_data, train_rul_data, val_data, val_rul_data, test_data, test_rul_data
