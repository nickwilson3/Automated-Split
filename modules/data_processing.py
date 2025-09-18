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
        required_columns = ['Name', 'Final_Leaning']
    
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


def check_sequential_optimization_feasibility(df, num_juries, jury_size):
    """
    Check if sequential jury optimization is feasible with the given parameters.
    For sequential optimization, we only need to ensure we have enough total jurors.
    
    Parameters:
    df (pandas.DataFrame): Juror dataframe
    num_juries (int): Number of juries to form
    jury_size (int): Size of each jury
    
    Returns:
    tuple: (is_feasible (bool), message (str), feasibility_info (dict))
    """
    total_jurors_needed = num_juries * jury_size
    total_jurors_available = len(df)
    
    # Initialize feasibility info dictionary
    feasibility_info = {
        'total_jurors_available': total_jurors_available,
        'total_jurors_needed': total_jurors_needed,
        'can_fill_all_juries': total_jurors_available >= total_jurors_needed,
        'excess_jurors': max(0, total_jurors_available - total_jurors_needed),
        'complete_juries_possible': min(num_juries, total_jurors_available // jury_size),
        'sequential_approach': True
    }
    
    # Basic feasibility check
    if total_jurors_available < jury_size:
        return False, f"Insufficient jurors: Need at least {jury_size} to fill one jury but only have {total_jurors_available}", feasibility_info
    
    # Calculate demographic distributions for information
    if 'Final_Leaning' in df.columns:
        # Count overall P and D
        p_overall_count = df['Final_Leaning'].isin(['P', 'P+']).sum()
        d_overall_count = df['Final_Leaning'].isin(['D', 'D+']).sum()
        
        # Count granular leanings
        granular_counts = {
            'P+': (df['Final_Leaning'] == 'P+').sum(),
            'P': (df['Final_Leaning'] == 'P').sum(),
            'D': (df['Final_Leaning'] == 'D').sum(),
            'D+': (df['Final_Leaning'] == 'D+').sum()
        }
        
        # Store counts in feasibility info
        feasibility_info['total_p_overall'] = p_overall_count
        feasibility_info['total_d_overall'] = d_overall_count
        feasibility_info['granular_counts'] = granular_counts
        
        # Calculate P/D balance potential
        ideal_p_per_jury = jury_size // 2
        ideal_d_per_jury = jury_size - ideal_p_per_jury
        
        max_balanced_juries_p = p_overall_count // ideal_p_per_jury if ideal_p_per_jury > 0 else num_juries
        max_balanced_juries_d = d_overall_count // ideal_d_per_jury if ideal_d_per_jury > 0 else num_juries
        
        feasibility_info['max_balanced_juries'] = min(max_balanced_juries_p, max_balanced_juries_d, feasibility_info['complete_juries_possible'])
        
        print(f"P/D Balance Analysis:")
        print(f"  - Available: {p_overall_count} P-leaning, {d_overall_count} D-leaning")
        print(f"  - Ideal per jury: {ideal_p_per_jury} P, {ideal_d_per_jury} D")
        print(f"  - Maximum balanced juries possible: {feasibility_info['max_balanced_juries']}")
        
        print(f"Granular Leaning Analysis:")
        for category, count in granular_counts.items():
            print(f"  - {category}: {count} available")
    
    # Gender analysis
    if 'Gender' in df.columns:
        male_count = df['Gender'].isin(['M', 'Male', 'male', 'MALE']).sum()
        female_count = df['Gender'].isin(['F', 'Female', 'female', 'FEMALE']).sum()
        
        feasibility_info['total_male'] = male_count
        feasibility_info['total_female'] = female_count
        
        ideal_male_per_jury = jury_size // 2
        ideal_female_per_jury = jury_size - ideal_male_per_jury
        
        max_gender_balanced_juries = min(
            male_count // ideal_male_per_jury if ideal_male_per_jury > 0 else num_juries,
            female_count // ideal_female_per_jury if ideal_female_per_jury > 0 else num_juries,
            feasibility_info['complete_juries_possible']
        )
        
        feasibility_info['max_gender_balanced_juries'] = max_gender_balanced_juries
        
        print(f"Gender Balance Analysis:")
        print(f"  - Available: {male_count} male, {female_count} female")
        print(f"  - Maximum gender-balanced juries possible: {max_gender_balanced_juries}")
    
    # Determine feasibility message
    messages = []
    
    if feasibility_info['can_fill_all_juries']:
        messages.append(f"Sequential optimization feasible: Can fill all {num_juries} juries with {jury_size} jurors each")
    else:
        complete_juries = feasibility_info['complete_juries_possible']
        messages.append(f"Partial filling: Can fill {complete_juries} complete juries out of {num_juries} requested")
    
    if 'max_balanced_juries' in feasibility_info:
        balanced_juries = feasibility_info['max_balanced_juries']
        if balanced_juries < feasibility_info['complete_juries_possible']:
            messages.append(f"Perfect P/D balance achievable for {balanced_juries} juries; remaining juries may be imbalanced")
        else:
            messages.append(f"Perfect P/D balance potentially achievable for all fillable juries")
    
    # Sequential optimization is generally feasible if we can fill at least one jury
    is_feasible = feasibility_info['complete_juries_possible'] >= 1
    final_message = "; ".join(messages)
    
    return is_feasible, final_message, feasibility_info


def estimate_jury_balance_quality(df, num_juries, jury_size):
    """
    Estimate the expected balance quality for sequential optimization.
    Provides insight into what balance levels are achievable.
    
    Parameters:
    df (pandas.DataFrame): Juror dataframe
    num_juries (int): Number of juries to form
    jury_size (int): Size of each jury
    
    Returns:
    dict: Balance quality estimates
    """
    quality_estimates = {
        'sequential_predictions': {},
        'overall_quality': 'unknown'
    }
    
    total_jurors = len(df)
    fillable_juries = min(num_juries, total_jurors // jury_size)
    
    if 'Final_Leaning' in df.columns:
        p_total = df['Final_Leaning'].isin(['P', 'P+']).sum()
        d_total = df['Final_Leaning'].isin(['D', 'D+']).sum()
        
        # Estimate balance for each jury in sequence
        remaining_p = p_total
        remaining_d = d_total
        remaining_total = total_jurors
        
        for jury_num in range(1, fillable_juries + 1):
            ideal_p = jury_size // 2
            ideal_d = jury_size - ideal_p
            
            # Estimate what this jury can achieve
            available_p = min(remaining_p, ideal_p)
            available_d = min(remaining_d, ideal_d)
            
            # Check if perfect balance is possible
            if available_p == ideal_p and available_d == ideal_d:
                balance_quality = 'perfect'
            elif available_p + available_d == jury_size:
                balance_quality = 'complete_but_imbalanced'
            else:
                balance_quality = 'incomplete'
            
            quality_estimates['sequential_predictions'][f'jury_{jury_num}'] = {
                'predicted_p': available_p,
                'predicted_d': available_d,
                'predicted_balance_quality': balance_quality
            }
            
            # Update remaining counts (simulate assignment)
            remaining_p -= available_p
            remaining_d -= available_d
            remaining_total -= jury_size
    
    # Determine overall quality
    if fillable_juries == num_juries:
        perfect_juries = sum(1 for pred in quality_estimates['sequential_predictions'].values() 
                           if pred['predicted_balance_quality'] == 'perfect')
        if perfect_juries == num_juries:
            quality_estimates['overall_quality'] = 'excellent'
        elif perfect_juries >= num_juries // 2:
            quality_estimates['overall_quality'] = 'good'
        else:
            quality_estimates['overall_quality'] = 'fair'
    else:
        quality_estimates['overall_quality'] = 'limited'
    
    return quality_estimates


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
    
    # Add leaning summary if present (including granular)
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
    Adapted for sequential optimization approach.
    
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
        demographic_vars = ['Final_Leaning', 'Gender', 'Race', 'Age', 'Education', 'Marital']
    
    # Load data
    df = load_juror_data(file_path)
    
    # Validate data
    is_valid, message, df = validate_juror_data(df, drop_missing=drop_missing)
    if not is_valid:
        raise ValueError(message)
    else:
        print(message)  # Print message about dropped rows if any
    
    # Check feasibility for sequential optimization
    feasibility_result = check_sequential_optimization_feasibility(df, num_juries, jury_size)
    
    # Unpack the feasibility result
    is_feasible, feasibility_message, feasibility_info = feasibility_result
    
    if not is_feasible:
        raise ValueError(feasibility_message)
    
    # Generate balance quality estimates
    quality_estimates = estimate_jury_balance_quality(df, num_juries, jury_size)
    
    # Print sequential optimization analysis
    print("=== SEQUENTIAL OPTIMIZATION ANALYSIS ===")
    print(f"Total jurors available: {feasibility_info['total_jurors_available']}")
    print(f"Total jurors needed: {feasibility_info['total_jurors_needed']}")
    print(f"Complete juries possible: {feasibility_info['complete_juries_possible']} out of {num_juries}")
    
    if feasibility_info.get('excess_jurors', 0) > 0:
        print(f"Excess jurors: {feasibility_info['excess_jurors']} will remain unassigned")
    
    # Print balance predictions
    if 'sequential_predictions' in quality_estimates:
        print(f"Balance Quality Predictions:")
        for jury_key, prediction in quality_estimates['sequential_predictions'].items():
            jury_num = jury_key.split('_')[1]
            print(f"  - Jury {jury_num}: {prediction['predicted_p']}P/{prediction['predicted_d']}D ({prediction['predicted_balance_quality']})")
        
        print(f"Overall expected quality: {quality_estimates['overall_quality']}")
    
    # Summarize data
    summary = summarize_juror_data(df)
    
    # Prepare for optimization
    prepared_df, encodings, category_counts = prepare_data_for_optimization(df, demographic_vars)
    
    # Return all processed data with sequential optimization info
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
        'feasibility_info': feasibility_info,  # Sequential-specific feasibility info
        'quality_estimates': quality_estimates,  # Balance quality predictions
        'optimization_approach': 'sequential',
        'will_have_unassigned': feasibility_info['excess_jurors'] > 0
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
        
        # Print feasibility info
        feasibility_info = result['feasibility_info']
        print(f"Can fill all juries: {feasibility_info['can_fill_all_juries']}")
        print(f"Complete juries possible: {feasibility_info['complete_juries_possible']}")
        
        # Print quality estimates
        quality_estimates = result['quality_estimates']
        print(f"Expected overall quality: {quality_estimates['overall_quality']}")
        
        if 'sequential_predictions' in quality_estimates:
            print("Sequential balance predictions:")
            for jury_key, prediction in quality_estimates['sequential_predictions'].items():
                print(f"  {jury_key}: {prediction}")
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")