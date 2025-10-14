"""
Data processing module for simultaneous jury optimization.
Handles data loading, validation, feasibility analysis, and preparation.
"""

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


def check_simultaneous_optimization_feasibility(df, num_juries, jury_size):
    """
    Check if simultaneous jury optimization is feasible with the given parameters.
    Uses maximin logic to calculate achievable balance targets.
    
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
        'optimization_approach': 'simultaneous'
    }
    
    # Basic feasibility check
    if total_jurors_available < total_jurors_needed:
        return False, f"Insufficient jurors: Need {total_jurors_needed} but only have {total_jurors_available}", feasibility_info
    
    # Calculate demographic distributions and maximin targets
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
        
        # Calculate maximin targets for P/D balance
        ideal_p_per_jury = jury_size // 2
        ideal_d_per_jury = jury_size - ideal_p_per_jury
        
        # Check if we can achieve ideal balance
        total_p_needed = num_juries * ideal_p_per_jury
        total_d_needed = num_juries * ideal_d_per_jury
        
        can_achieve_ideal_p = p_overall_count >= total_p_needed
        can_achieve_ideal_d = d_overall_count >= total_d_needed
        
        if can_achieve_ideal_p and can_achieve_ideal_d:
            # Perfect balance is achievable
            p_target_per_jury = ideal_p_per_jury
            d_target_per_jury = ideal_d_per_jury
            balance_status = 'ideal_achievable'
        else:
            # Use maximin to distribute as evenly as possible
            p_target_per_jury = p_overall_count // num_juries
            d_target_per_jury = d_overall_count // num_juries
            balance_status = 'maximin_distribution'
        
        feasibility_info['p_target_per_jury'] = p_target_per_jury
        feasibility_info['d_target_per_jury'] = d_target_per_jury
        feasibility_info['balance_status'] = balance_status
        
        print(f"\nP/D Balance Analysis (Simultaneous Optimization):")
        print(f"  - Available: {p_overall_count} P-leaning, {d_overall_count} D-leaning")
        print(f"  - Ideal per jury: {ideal_p_per_jury} P, {ideal_d_per_jury} D")
        print(f"  - Achievable target per jury: {p_target_per_jury} P, {d_target_per_jury} D")
        print(f"  - Balance status: {balance_status}")
        
        # Calculate maximin targets for granular leanings
        granular_targets = {}
        for category in ['P+', 'P', 'D', 'D+']:
            available = granular_counts[category]
            maximin_target = available // num_juries
            remainder = available % num_juries
            granular_targets[category] = maximin_target
            
            if remainder > 0:
                print(f"  - {category}: {maximin_target} per jury (with {remainder} extra to distribute)")
            else:
                print(f"  - {category}: {maximin_target} per jury (even distribution)")
        
        feasibility_info['granular_targets'] = granular_targets
        
        print(f"\nGranular Leaning Targets (Maximin Distribution):")
        for category, target in granular_targets.items():
            print(f"  - {category}: {target} per jury (total available: {granular_counts[category]})")
    
    # Gender analysis
    if 'Gender' in df.columns:
        male_count = df['Gender'].isin(['M', 'Male', 'male', 'MALE']).sum()
        female_count = df['Gender'].isin(['F', 'Female', 'female', 'FEMALE']).sum()
        
        feasibility_info['total_male'] = male_count
        feasibility_info['total_female'] = female_count
        
        # Ideal targets for gender
        ideal_male_per_jury = jury_size // 2
        ideal_female_per_jury = jury_size - ideal_male_per_jury
        
        # Maximin targets for gender
        male_target_per_jury = male_count // num_juries
        female_target_per_jury = female_count // num_juries
        
        total_male_needed = num_juries * ideal_male_per_jury
        total_female_needed = num_juries * ideal_female_per_jury
        
        can_achieve_ideal_gender = (male_count >= total_male_needed and 
                                     female_count >= total_female_needed)
        
        if can_achieve_ideal_gender:
            gender_status = 'ideal_achievable'
        else:
            gender_status = 'maximin_distribution'
        
        feasibility_info['male_target_per_jury'] = male_target_per_jury
        feasibility_info['female_target_per_jury'] = female_target_per_jury
        feasibility_info['gender_status'] = gender_status
        
        print(f"\nGender Balance Analysis:")
        print(f"  - Available: {male_count} male, {female_count} female")
        print(f"  - Ideal per jury: {ideal_male_per_jury} M, {ideal_female_per_jury} F")
        print(f"  - Maximin target per jury: {male_target_per_jury} M, {female_target_per_jury} F")
        print(f"  - Gender status: {gender_status}")
    
    # Race analysis
    if 'Race' in df.columns:
        race_counts = df['Race'].value_counts().to_dict()
        feasibility_info['race_counts'] = race_counts
        
        print(f"\nRace Distribution:")
        for race, count in race_counts.items():
            target_per_jury = count // num_juries
            print(f"  - {race}: {count} total ({target_per_jury} per jury using maximin)")
    
    # Age analysis
    if 'Age' in df.columns:
        # Create age groups if not already present
        if 'AgeGroup' not in df.columns:
            age_bins = [0, 30, 40, 50, 60, 100]
            age_labels = ['<30', '30-39', '40-49', '50-59', '60+']
            df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
        
        age_counts = df['AgeGroup'].value_counts().to_dict()
        feasibility_info['age_counts'] = age_counts
        
        print(f"\nAge Group Distribution:")
        for age, count in age_counts.items():
            target_per_jury = count // num_juries
            print(f"  - {age}: {count} total ({target_per_jury} per jury using maximin)")
    
    # Education analysis
    if 'Education' in df.columns:
        education_counts = df['Education'].value_counts().to_dict()
        feasibility_info['education_counts'] = education_counts
        
        print(f"\nEducation Distribution:")
        for edu, count in education_counts.items():
            target_per_jury = count // num_juries
            print(f"  - {edu}: {count} total ({target_per_jury} per jury using maximin)")
    
    # Marital status analysis
    if 'Marital' in df.columns:
        marital_counts = df['Marital'].value_counts().to_dict()
        feasibility_info['marital_counts'] = marital_counts
        
        print(f"\nMarital Status Distribution:")
        for marital, count in marital_counts.items():
            target_per_jury = count // num_juries
            print(f"  - {marital}: {count} total ({target_per_jury} per jury using maximin)")
    
    # Determine feasibility message
    messages = []
    
    if feasibility_info['can_fill_all_juries']:
        messages.append(f"Simultaneous optimization feasible: Can fill all {num_juries} juries with {jury_size} jurors each")
    else:
        messages.append(f"Insufficient jurors to fill all {num_juries} juries")
    
    if 'balance_status' in feasibility_info:
        if feasibility_info['balance_status'] == 'ideal_achievable':
            messages.append(f"Ideal {ideal_p_per_jury}/{ideal_d_per_jury} P/D balance achievable in all juries")
        else:
            messages.append(f"Using maximin distribution: {p_target_per_jury}/{d_target_per_jury} P/D per jury")
    
    if 'gender_status' in feasibility_info:
        if feasibility_info['gender_status'] == 'ideal_achievable':
            messages.append(f"Ideal gender balance achievable in all juries")
        else:
            messages.append(f"Using maximin for gender: {male_target_per_jury}M/{female_target_per_jury}F per jury")
    
    is_feasible = feasibility_info['can_fill_all_juries']
    final_message = "; ".join(messages)
    
    return is_feasible, final_message, feasibility_info


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
        summary['leaning_percentages'] = (df['Final_Leaning'].value_counts(normalize=True) * 100).round(2).to_dict()
    
    # Add gender summary if present
    if 'Gender' in df.columns:
        summary['gender_counts'] = df['Gender'].value_counts().to_dict()
        summary['gender_percentages'] = (df['Gender'].value_counts(normalize=True) * 100).round(2).to_dict()
    
    # Add race summary if present
    if 'Race' in df.columns:
        summary['race_counts'] = df['Race'].value_counts().to_dict()
        summary['race_percentages'] = (df['Race'].value_counts(normalize=True) * 100).round(2).to_dict()
    
    # Add age summary if present
    if 'Age' in df.columns:
        summary['age_stats'] = {
            'mean': round(df['Age'].mean(), 1),
            'median': df['Age'].median(),
            'min': df['Age'].min(),
            'max': df['Age'].max()
        }
        
        # Add age group counts (create age buckets if not present)
        if 'AgeGroup' not in df.columns:
            age_bins = [0, 30, 40, 50, 60, 100]
            age_labels = ['<30', '30-39', '40-49', '50-59', '60+']
            df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
        
        summary['age_group_counts'] = df['AgeGroup'].value_counts().to_dict()
    
    # Add education summary if present
    if 'Education' in df.columns:
        summary['education_counts'] = df['Education'].value_counts().to_dict()
        summary['education_percentages'] = (df['Education'].value_counts(normalize=True) * 100).round(2).to_dict()
    
    # Add marital status summary if present
    if 'Marital' in df.columns:
        summary['marital_status_counts'] = df['Marital'].value_counts().to_dict()
        summary['marital_status_percentages'] = (df['Marital'].value_counts(normalize=True) * 100).round(2).to_dict()
    
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
            # For categorical variables, store category information
            categories = prepared_df[var].dropna().unique()
            category_counts[var] = {cat: (prepared_df[var] == cat).sum() for cat in categories}
            
            # Store encoding mapping
            encodings[var] = {i: cat for i, cat in enumerate(categories)}
        
        elif var == 'Age' and 'Age' in prepared_df.columns:
            # For age, create age groups if not already present
            if 'AgeGroup' not in prepared_df.columns:
                age_bins = [0, 30, 40, 50, 60, 100]
                age_labels = ['<30', '30-39', '40-49', '50-59', '60+']
                prepared_df['AgeGroup'] = pd.cut(prepared_df['Age'], bins=age_bins, labels=age_labels, right=False)
            
            # Store age group counts
            categories = prepared_df['AgeGroup'].dropna().unique()
            category_counts['AgeGroup'] = {cat: (prepared_df['AgeGroup'] == cat).sum() for cat in categories}
            encodings['AgeGroup'] = {i: cat for i, cat in enumerate(categories)}
    
    return prepared_df, encodings, category_counts


def process_juror_data(file_path, num_juries, jury_size, demographic_vars=None, drop_missing=True):
    """
    Main function to process juror data from file to optimization-ready format.
    Adapted for simultaneous optimization approach.
    
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
    
    print("=" * 70)
    print("PROCESSING JUROR DATA FOR SIMULTANEOUS OPTIMIZATION")
    print("=" * 70)
    
    # Load data
    df = load_juror_data(file_path)
    
    # Validate data
    is_valid, message, df = validate_juror_data(df, drop_missing=drop_missing)
    if not is_valid:
        raise ValueError(message)
    else:
        if "Dropped" in message:
            print(f"⚠ {message}")
        else:
            print(f"✓ {message}")
    
    # Check feasibility for simultaneous optimization
    feasibility_result = check_simultaneous_optimization_feasibility(df, num_juries, jury_size)
    
    # Unpack the feasibility result
    is_feasible, feasibility_message, feasibility_info = feasibility_result
    
    if not is_feasible:
        raise ValueError(feasibility_message)
    
    print("\n" + "=" * 70)
    print("SIMULTANEOUS OPTIMIZATION FEASIBILITY SUMMARY")
    print("=" * 70)
    print(f"Total jurors available: {feasibility_info['total_jurors_available']}")
    print(f"Total jurors needed: {feasibility_info['total_jurors_needed']}")
    print(f"Can fill all juries: {'✓ YES' if feasibility_info['can_fill_all_juries'] else '✗ NO'}")
    
    if feasibility_info.get('excess_jurors', 0) > 0:
        print(f"Excess jurors: {feasibility_info['excess_jurors']} will remain unassigned")
    
    print(f"\nOptimization approach: All {num_juries} juries optimized simultaneously")
    print(f"Goal: Maximize similarity between all jury pairs")
    print("=" * 70)
    
    # Summarize data
    summary = summarize_juror_data(df)
    
    # Prepare for optimization
    prepared_df, encodings, category_counts = prepare_data_for_optimization(df, demographic_vars)
    
    # Return all processed data with simultaneous optimization info
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
        'feasibility_info': feasibility_info,
        'optimization_approach': 'simultaneous',
        'will_have_unassigned': feasibility_info['excess_jurors'] > 0
    }


if __name__ == "__main__":
    # Example usage (for testing)
    print("Testing data_processing module...")
    
    file_path = r"C:\Users\NicholasWilson\OneDrive - Trial Behavior Consulting\City_Split.xlsx"  # UPDATE THIS
    num_juries = 2
    jury_size = 12
    
    try:
        result = process_juror_data(file_path, num_juries, jury_size)
        
        print("\n" + "=" * 70)
        print("DATA PROCESSING TEST COMPLETE")
        print("=" * 70)
        print(f"✓ Data processed successfully")
        print(f"✓ Found {result['summary']['total_jurors']} jurors")
        print(f"✓ Feasibility check passed")
        
        # Print some summary information
        if 'leaning_counts' in result['summary']:
            print(f"\nLeaning distribution:")
            for leaning, count in result['summary']['leaning_counts'].items():
                pct = result['summary']['leaning_percentages'][leaning]
                print(f"  - {leaning}: {count} ({pct}%)")
        
        # Print feasibility info
        feasibility_info = result['feasibility_info']
        print(f"\nFeasibility:")
        print(f"  - Can fill all juries: {feasibility_info['can_fill_all_juries']}")
        print(f"  - Balance status: {feasibility_info.get('balance_status', 'N/A')}")
        print(f"  - P target per jury: {feasibility_info.get('p_target_per_jury', 'N/A')}")
        print(f"  - D target per jury: {feasibility_info.get('d_target_per_jury', 'N/A')}")
        
        print("\n✓ Module test passed!")
        
    except FileNotFoundError:
        print("\n✗ Error: File not found. Please update the file_path variable.")
    except Exception as e:
        print(f"\n✗ Error processing data: {str(e)}")
        import traceback
        traceback.print_exc()