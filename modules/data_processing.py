# data_processing.py
import pandas as pd
import numpy as np


def load_juror_data(file_path):
    """
    Load juror data from an Excel file.
    
    Parameters:
    file_path (str): Path to the Excel file containing juror data
    
    Returns:
    pandas.DataFrame: Loaded juror dataframe
    """
    try:
        df = pd.read_excel(file_path)
        print(f"Successfully loaded data with {len(df)} jurors")
        return df
    except Exception as e:
        raise Exception(f"Error loading juror data: {str(e)}")


def validate_juror_data(df, required_columns=None, drop_missing=False):
    """
    Validate the juror data for required columns and check for missing values.
    
    Parameters:
    df (pandas.DataFrame): Juror dataframe
    required_columns (list, optional): List of columns that must be present
    drop_missing (bool): If True, drop rows with missing values instead of failing
    
    Returns:
    tuple: (is_valid (bool), message (str), cleaned_df (DataFrame))
    """
    if required_columns is None:
        required_columns = ['Name', 'Final_Leaning']  # Updated to match your column name
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}", df
    
    # Check for missing values in required columns
    missing_values = df[required_columns].isnull().sum()
    has_missing = missing_values.sum() > 0
    
    if has_missing:
        if drop_missing:
            # Drop rows with missing values in required columns
            original_count = len(df)
            cleaned_df = df.dropna(subset=required_columns)
            dropped_count = original_count - len(cleaned_df)
            return True, f"Dropped {dropped_count} rows with missing values in required columns", cleaned_df
        else:
            missing_cols = missing_values[missing_values > 0].index.tolist()
            return False, f"Missing values in critical columns: {', '.join(missing_cols)}", df
    
    return True, "Data validation passed", df


def check_optimization_feasibility(df, num_juries, jury_size):
    """
    Check if the jury optimization problem is feasible with the given parameters.
    Now includes enhanced gender balance checking.
    
    Parameters:
    df (pandas.DataFrame): Juror dataframe
    num_juries (int): Number of juries to form
    jury_size (int): Size of each jury
    
    Returns:
    tuple: (is_feasible (bool), message (str), balance_info (dict))
    """
    total_jurors_needed = num_juries * jury_size
    total_jurors_available = len(df)
    
    # Initialize comprehensive balance info dictionary
    balance_info = {
        # P/D balance info
        'has_enough_p': True,
        'has_enough_d': True,
        'total_p': 0,
        'total_d': 0,
        'optimal_p_distribution': None,
        'optimal_d_distribution': None,
        'ideal_p_per_jury': None,
        'ideal_d_per_jury': None,
        
        # Gender balance info (new)
        'has_enough_male': True,
        'has_enough_female': True,
        'total_male': 0,
        'total_female': 0,
        'optimal_male_distribution': None,
        'optimal_female_distribution': None,
        'gender_balance_possible': True,
        
        # General info
        'unassigned_count': max(0, total_jurors_available - total_jurors_needed),
        'can_fill_all_juries': total_jurors_available >= total_jurors_needed
    }
    
    # Basic feasibility check
    if total_jurors_needed > total_jurors_available:
        return False, f"Insufficient jurors: Need {total_jurors_needed} but only have {total_jurors_available}", balance_info
    
    # P/D leaning balance checks
    if 'Final_Leaning' in df.columns:
        p_count = df['Final_Leaning'].isin(['P', 'P+']).sum()
        d_count = df['Final_Leaning'].isin(['D', 'D+']).sum()
        
        # Store counts in balance info
        balance_info['total_p'] = p_count
        balance_info['total_d'] = d_count
        
        # Check if perfect balance is possible
        if jury_size % 2 == 0:  # Even jury size
            ideal_p_per_jury = jury_size // 2
            ideal_d_per_jury = jury_size // 2
            
            balance_info['ideal_p_per_jury'] = ideal_p_per_jury
            balance_info['ideal_d_per_jury'] = ideal_d_per_jury
            
            # Check if we have enough P jurors
            if p_count < ideal_p_per_jury * num_juries:
                balance_info['has_enough_p'] = False
                # Calculate optimal P distribution using maximin approach
                balance_info['optimal_p_distribution'] = calculate_optimal_distribution(p_count, num_juries)
            
            # Check if we have enough D jurors
            if d_count < ideal_d_per_jury * num_juries:
                balance_info['has_enough_d'] = False
                # Calculate optimal D distribution using maximin approach
                balance_info['optimal_d_distribution'] = calculate_optimal_distribution(d_count, num_juries)
                
            # If both are sufficient, set perfect distributions
            if balance_info['has_enough_p'] and balance_info['has_enough_d']:
                balance_info['optimal_p_distribution'] = [ideal_p_per_jury] * num_juries
                balance_info['optimal_d_distribution'] = [ideal_d_per_jury] * num_juries
        
        # For odd jury size, we'll need to decide which juries get the extra P or D
        else:
            # For odd sizes, one side gets the extra juror
            base_p = jury_size // 2
            base_d = jury_size // 2
            
            # Check if we can achieve near-perfect balance
            if p_count >= base_p * num_juries and d_count >= base_d * num_juries:
                # We can achieve good balance, calculate optimal distribution
                balance_info['optimal_p_distribution'] = calculate_optimal_distribution(
                    min(p_count, num_juries * (base_p + 1)), num_juries
                )
                balance_info['optimal_d_distribution'] = [
                    jury_size - p for p in balance_info['optimal_p_distribution']
                ]
            else:
                # Not enough for good balance, use maximin
                if p_count < base_p * num_juries:
                    balance_info['has_enough_p'] = False
                    balance_info['optimal_p_distribution'] = calculate_optimal_distribution(p_count, num_juries)
                if d_count < base_d * num_juries:
                    balance_info['has_enough_d'] = False
                    balance_info['optimal_d_distribution'] = calculate_optimal_distribution(d_count, num_juries)
    
    # Gender balance checks (new logic)
    if 'Gender' in df.columns:
        # Count available male and female jurors
        male_count = df['Gender'].isin(['M', 'Male', 'male', 'MALE']).sum()
        female_count = df['Gender'].isin(['F', 'Female', 'female', 'FEMALE']).sum()
        
        balance_info['total_male'] = male_count
        balance_info['total_female'] = female_count
        
        # Calculate optimal gender distribution GIVEN the leaning constraints
        optimal_gender_distribution = calculate_optimal_gender_distribution(
            balance_info, male_count, female_count, num_juries, jury_size
        )
        
        balance_info.update(optimal_gender_distribution)
    
    # Determine overall feasibility message
    messages = []
    if not balance_info['has_enough_p']:
        messages.append(f"Warning: Not enough 'P' jurors for perfect balance. Have {balance_info['total_p']}, optimal distribution calculated.")
    if not balance_info['has_enough_d']:
        messages.append(f"Warning: Not enough 'D' jurors for perfect balance. Have {balance_info['total_d']}, optimal distribution calculated.")
    if not balance_info['gender_balance_possible']:
        messages.append(f"Warning: Perfect gender balance not possible given leaning constraints.")
    
    final_message = "; ".join(messages) if messages else "Problem appears feasible with good balance potential"
    
    return True, final_message, balance_info


def calculate_optimal_distribution(total_count, num_groups):
    """
    Calculate the optimal distribution of limited resources across groups
    using a maximin approach (maximizing the minimum allocation).
    
    Parameters:
    total_count (int): Total number of resources to distribute
    num_groups (int): Number of groups to distribute across
    
    Returns:
    list: Optimal distribution with the highest possible minimum value
    """
    # Base distribution - start by dividing evenly
    base_value = total_count // num_groups
    
    # Calculate how many groups need to get an extra resource
    remainder = total_count % num_groups
    
    # Create the distribution: some groups get (base_value + 1), the rest get base_value
    distribution = [base_value + 1] * remainder + [base_value] * (num_groups - remainder)
    
    return distribution


def calculate_optimal_gender_distribution(balance_info, male_count, female_count, num_juries, jury_size):
    """
    Calculate optimal gender distribution given the P/D leaning constraints.
    This is the key new function for hierarchical optimization.
    
    Parameters:
    balance_info (dict): Contains P/D distribution information
    male_count (int): Total male jurors available
    female_count (int): Total female jurors available  
    num_juries (int): Number of juries to form
    jury_size (int): Size of each jury
    
    Returns:
    dict: Gender distribution information to add to balance_info
    """
    gender_info = {
        'has_enough_male': True,
        'has_enough_female': True,
        'optimal_male_distribution': None,
        'optimal_female_distribution': None,
        'gender_balance_possible': True,
        'gender_targets_per_jury': []
    }
    
    # If we don't have P/D distributions, fall back to simple gender balance
    if not balance_info.get('optimal_p_distribution') and not balance_info.get('optimal_d_distribution'):
        # Simple case: just try to balance gender across all juries
        ideal_male_per_jury = jury_size // 2
        ideal_female_per_jury = jury_size - ideal_male_per_jury
        
        if male_count >= ideal_male_per_jury * num_juries:
            gender_info['optimal_male_distribution'] = [ideal_male_per_jury] * num_juries
        else:
            gender_info['has_enough_male'] = False
            gender_info['optimal_male_distribution'] = calculate_optimal_distribution(male_count, num_juries)
            
        if female_count >= ideal_female_per_jury * num_juries:
            gender_info['optimal_female_distribution'] = [ideal_female_per_jury] * num_juries
        else:
            gender_info['has_enough_female'] = False
            gender_info['optimal_female_distribution'] = calculate_optimal_distribution(female_count, num_juries)
            
        # Check if perfect gender balance is possible
        total_assigned = sum(gender_info['optimal_male_distribution']) + sum(gender_info['optimal_female_distribution'])
        if total_assigned < num_juries * jury_size:
            gender_info['gender_balance_possible'] = False
            
        return gender_info
    
    # Complex case: we have P/D constraints, need to find best gender balance within those constraints
    # This is where the hierarchical optimization logic would go
    
    # For now, we'll implement a simplified version that tries to balance gender
    # while respecting that each jury needs to maintain its P/D target
    
    # Start with the assumption that we'll try to get as close to 50/50 gender as possible
    # within each jury, given the size constraints
    optimal_male_dist = []
    optimal_female_dist = []
    
    for jury_idx in range(num_juries):
        # For each jury, try to get as close to 50/50 gender as possible
        target_male = jury_size // 2
        target_female = jury_size - target_male
        
        optimal_male_dist.append(target_male)
        optimal_female_dist.append(target_female)
    
    # Check if we have enough of each gender
    total_male_needed = sum(optimal_male_dist)
    total_female_needed = sum(optimal_female_dist)
    
    if male_count < total_male_needed:
        gender_info['has_enough_male'] = False
        gender_info['optimal_male_distribution'] = calculate_optimal_distribution(male_count, num_juries)
        # Recalculate female distribution based on remaining spots
        gender_info['optimal_female_distribution'] = [
            jury_size - m for m in gender_info['optimal_male_distribution']
        ]
    elif female_count < total_female_needed:
        gender_info['has_enough_female'] = False
        gender_info['optimal_female_distribution'] = calculate_optimal_distribution(female_count, num_juries)
        # Recalculate male distribution based on remaining spots
        gender_info['optimal_male_distribution'] = [
            jury_size - f for f in gender_info['optimal_female_distribution']
        ]
    else:
        # We have enough of both genders
        gender_info['optimal_male_distribution'] = optimal_male_dist
        gender_info['optimal_female_distribution'] = optimal_female_dist
    
    # Check if the final distribution is feasible
    total_gender_assigned = sum(gender_info['optimal_male_distribution']) + sum(gender_info['optimal_female_distribution'])
    if total_gender_assigned != num_juries * jury_size:
        gender_info['gender_balance_possible'] = False
    
    return gender_info


def summarize_juror_data(df):
    """
    Generate summary statistics for the juror data.
    
    Parameters:
    df (pandas.DataFrame): Juror dataframe
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        'total_jurors': len(df)
    }
    
    # Add leaning summary if present
    if 'Final_Leaning' in df.columns:
        summary['leaning_counts'] = df['Final_Leaning'].value_counts().to_dict()
        summary['leaning_percentages'] = (df['Final_Leaning'].value_counts(normalize=True) * 100).to_dict()
    
    # Add gender summary if present
    if 'Gender' in df.columns:
        summary['gender_counts'] = df['Gender'].value_counts().to_dict()
        summary['gender_percentages'] = (df['Gender'].value_counts(normalize=True) * 100).to_dict()
    
    # Add race summary if present
    if 'Race' in df.columns:
        summary['race_counts'] = df['Race'].value_counts().to_dict()
        summary['race_percentages'] = (df['Race'].value_counts(normalize=True) * 100).to_dict()
    
    # Add age summary if present
    if 'Age' in df.columns:
        summary['age_stats'] = {
            'mean': df['Age'].mean(),
            'median': df['Age'].median(),
            'min': df['Age'].min(),
            'max': df['Age'].max()
        }
        
        # Add age group counts (create age buckets)
        age_bins = [0, 30, 40, 50, 60, 100]
        age_labels = ['<30', '30-39', '40-49', '50-59', '60+']
        df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
        summary['age_group_counts'] = df['AgeGroup'].value_counts().to_dict()
    
    # Add education summary if present
    if 'Education' in df.columns:
        summary['education_counts'] = df['Education'].value_counts().to_dict()
    
    # Add marital status summary if present
    if 'Marital' in df.columns:
        summary['marital_status_counts'] = df['Marital'].value_counts().to_dict()
    
    return summary


def prepare_data_for_optimization(df, demographic_vars):
    """
    Prepare the juror data for optimization by encoding categorical variables
    and creating necessary data structures.
    
    Parameters:
    df (pandas.DataFrame): Juror dataframe
    demographic_vars (list): List of demographic variables to consider in optimization
    
    Returns:
    tuple: (prepared_df (DataFrame), encodings (dict), category_counts (dict))
    """
    # Create a copy to avoid modifying the original
    prepared_df = df.copy()
    
    # Dictionary to store encoding mappings
    encodings = {}
    
    # Dictionary to store category counts
    category_counts = {}
    
    # Process each demographic variable
    for var in demographic_vars:
        if var not in prepared_df.columns:
            continue
            
        if prepared_df[var].dtype == 'object' or prepared_df[var].dtype.name == 'category':
            # For categorical variables, create one-hot encoding
            categories = prepared_df[var].unique()
            category_counts[var] = {cat: (prepared_df[var] == cat).sum() for cat in categories}
            
            # Store encoding mapping
            encodings[var] = {i: cat for i, cat in enumerate(categories)}
            
            # Create dummy variables
            dummies = pd.get_dummies(prepared_df[var], prefix=var)
            prepared_df = pd.concat([prepared_df, dummies], axis=1)
        
        elif var == 'Age' and 'Age' in prepared_df.columns:
            # For age, create age groups if not already present
            if 'AgeGroup' not in prepared_df.columns:
                age_bins = [0, 30, 40, 50, 60, 100]
                age_labels = ['<30', '30-39', '40-49', '50-59', '60+']
                prepared_df['AgeGroup'] = pd.cut(prepared_df['Age'], bins=age_bins, labels=age_labels, right=False)
            
            # One-hot encode age groups
            age_dummies = pd.get_dummies(prepared_df['AgeGroup'], prefix='Age')
            prepared_df = pd.concat([prepared_df, age_dummies], axis=1)
            
            # Store age group counts
            categories = prepared_df['AgeGroup'].unique()
            category_counts['AgeGroup'] = {cat: (prepared_df['AgeGroup'] == cat).sum() for cat in categories}
            encodings['AgeGroup'] = {i: cat for i, cat in enumerate(categories)}
    
    return prepared_df, encodings, category_counts


def process_juror_data(file_path, num_juries, jury_size, demographic_vars=None, drop_missing=True):
    """
    Main function to process juror data from file to optimization-ready format.
    Enhanced with hierarchical balance checking.
    
    Parameters:
    file_path (str): Path to the Excel file containing juror data
    num_juries (int): Number of juries to form
    jury_size (int): Size of each jury
    demographic_vars (list, optional): List of demographic variables to consider
    drop_missing (bool): If True, drop rows with missing values instead of failing
    
    Returns:
    dict: Dictionary containing all processed data and information
    """
    if demographic_vars is None:
        # Fixed the missing comma bug!
        demographic_vars = ['Final_Leaning', 'Gender', 'Race', 'Age', 'Education', 'Marital']
    
    # Load data
    df = load_juror_data(file_path)
    
    # Validate data
    is_valid, message, df = validate_juror_data(df, drop_missing=drop_missing)
    if not is_valid:
        raise ValueError(message)
    else:
        print(message)  # Print message about dropped rows if any
    
    # Check feasibility with enhanced balance checking
    feasibility_result = check_optimization_feasibility(df, num_juries, jury_size)
    
    # Unpack the enhanced feasibility result
    is_feasible, feasibility_message, balance_info = feasibility_result
    
    if not is_feasible:
        raise ValueError(feasibility_message)
    
    # Add a note about unassigned jurors if applicable
    if balance_info['unassigned_count'] > 0:
        print(f"Note: {balance_info['unassigned_count']} jurors will be initially marked as unassigned")
    
    # Print balance information
    if not balance_info.get('gender_balance_possible', True):
        print("Note: Perfect gender balance may not be achievable given other constraints")
    
    # Summarize data
    summary = summarize_juror_data(df)
    
    # Prepare for optimization
    prepared_df, encodings, category_counts = prepare_data_for_optimization(df, demographic_vars)
    
    # Return all processed data with enhanced balance info
    return {
        'original_data': df,
        'prepared_data': prepared_df,
        'summary': summary,
        'encodings': encodings,
        'category_counts': category_counts,
        'num_juries': num_juries,
        'jury_size': jury_size,
        'demographic_vars': demographic_vars,
        'feasibility_message': feasibility_message,
        'balance_info': balance_info,  # Enhanced balance info including gender
        'will_have_unassigned': balance_info['unassigned_count'] > 0
    }


if __name__ == "__main__":
    # Example usage (for testing)
    file_path = r"C:\Users\NicholasWilson\OneDrive - Trial Behavior Consulting\AutoJurySplit_Data.xlsx"
    num_juries = 2
    jury_size = 12
    
    try:
        result = process_juror_data(file_path, num_juries, jury_size)
        print(f"Data processed successfully. Found {result['summary']['total_jurors']} jurors.")
        print(f"Feasibility check: {result['feasibility_message']}")
        
        # Print some summary information
        if 'leaning_counts' in result['summary']:
            print(f"Leaning distribution: {result['summary']['leaning_counts']}")
        
        # Print balance info
        balance_info = result['balance_info']
        print(f"P/D balance possible: P={balance_info['has_enough_p']}, D={balance_info['has_enough_d']}")
        print(f"Gender balance possible: {balance_info.get('gender_balance_possible', 'Unknown')}")
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")