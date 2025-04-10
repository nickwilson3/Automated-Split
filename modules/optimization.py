# optimization.py
import numpy as np
import pandas as pd
from pulp import *

def create_optimization_model(data_dict, priority_weights=None):
    """
    Create the Mixed Integer Programming model for jury assignment optimization.
    
    Parameters:
    data_dict (dict): Dictionary containing processed juror data and parameters
    priority_weights (dict, optional): Dictionary mapping demographic variables to their priority weights
    
    Returns:
    LpProblem: The optimization model
    dict: Dictionary containing model variables and other information
    """
    # Extract data from the dictionary
    df = data_dict['original_data']
    prepared_df = data_dict['prepared_data']
    num_juries = data_dict['num_juries']
    jury_size = data_dict['jury_size']
    category_counts = data_dict['category_counts']
    demographic_vars = data_dict['demographic_vars']
    
    # Extract P/D balance info if available
    p_d_balance_info = data_dict.get('p_d_balance_info', {
        'has_enough_p': True,
        'has_enough_d': True,
        'optimal_p_distribution': None,
        'optimal_d_distribution': None,
        'ideal_p_per_jury': jury_size // 2 if jury_size % 2 == 0 else None
    })
    
    # Set default priority weights if not provided
    if priority_weights is None:
        # Default priority: Leaning is highest priority, followed by others
        priority_weights = {
            'Final_Leaning': 5.0,
            'Gender': 4.0,
            'Race': 3.0,
            'AgeGroup': 2.0,
            'Education': 1.0,
            'Marital': 0.5
        }
    '''
    custom weights provided by user
    '''    
    
    # Create the optimization model
    model = LpProblem("Jury_Assignment_Optimization", LpMinimize)
    
    # STEP 2: Define decision variables
    # Binary variables x[i,j] = 1 if juror i is assigned to jury j
    juror_indices = df.index.tolist()
    jury_indices = range(1, num_juries + 1)
    
    x = LpVariable.dicts("Assign", 
                         [(i, j) for i in juror_indices for j in jury_indices],
                         cat=LpBinary)
    
    # Define deviation variables for each demographic attribute and jury
    # These will measure how far each jury deviates from the ideal distribution
    deviation_vars = {}
    
    # STEP 3: Define constraints
    # Constraint A: Each jury must have exactly jury_size jurors
    for j in jury_indices:
        model += lpSum(x[(i, j)] for i in juror_indices) == jury_size, f"Jury_Size_{j}"
    
    # Constraint B: Each juror must be assigned to exactly one jury
    for i in juror_indices:
        model += lpSum(x[(i, j)] for j in jury_indices) <= 1, f"Juror_Assignment_{i}"
    
    # Constraint C: Balance constraints for each demographic variable
    objective_terms = []
    
    # Process leaning balance first (highest priority)
    if 'Final_Leaning' in demographic_vars and 'Final_Leaning' in df.columns:
        p_indices = df[df['Final_Leaning'].isin(['P', 'P+'])].index.tolist()
        d_indices = df[df['Final_Leaning'].isin(['D', 'D+'])].index.tolist()
        
        # Create deviation variables for leaning
        p_plus_dev = LpVariable.dicts("P_Plus_Dev", jury_indices, lowBound=0)
        p_minus_dev = LpVariable.dicts("P_Minus_Dev", jury_indices, lowBound=0)
        
        # Handle case where we don't have enough P jurors for perfect balance
        if not p_d_balance_info['has_enough_p'] and p_d_balance_info['optimal_p_distribution']:
            # Use optimal distribution as targets
            p_targets = p_d_balance_info['optimal_p_distribution']
            
            for j in jury_indices:
                # Use the calculated target for this jury instead of ideal_p
                target_p_for_jury = p_targets[j-1]  # j is 1-indexed, list is 0-indexed
                
                # Count P leaning jurors in each jury
                p_count = lpSum(x[(i, j)] for i in p_indices)
                
                # Constraints for P leaning deviation
                model += p_count - p_plus_dev[j] + p_minus_dev[j] == target_p_for_jury, f"P_Balance_{j}"
                
                # Add to objective with high priority weight
                objective_terms.append(priority_weights['Final_Leaning'] * (p_plus_dev[j] + p_minus_dev[j]))
        
        # Handle case where we don't have enough D jurors for perfect balance
        elif not p_d_balance_info['has_enough_d'] and p_d_balance_info['optimal_d_distribution']:
            # Calculate the targets for P jurors (jury_size - D target)
            d_targets = p_d_balance_info['optimal_d_distribution']
            p_targets = [jury_size - d_targets[j-1] for j in jury_indices]
            
            for j in jury_indices:
                # Use the calculated target for this jury
                target_p_for_jury = p_targets[j-1]  # j is 1-indexed, list is 0-indexed
                
                # Count P leaning jurors in each jury
                p_count = lpSum(x[(i, j)] for i in p_indices)
                
                # Constraints for P leaning deviation
                model += p_count - p_plus_dev[j] + p_minus_dev[j] == target_p_for_jury, f"P_Balance_{j}"
                
                # Add to objective with high priority weight
                objective_terms.append(priority_weights['Final_Leaning'] * (p_plus_dev[j] + p_minus_dev[j]))
        
        # Default case: we have enough of both P and D jurors
        else:
            # Ideal distribution is 50% P and 50% D if jury_size is even
            ideal_p = jury_size // 2
            
            for j in jury_indices:
                # Count P leaning jurors in each jury
                p_count = lpSum(x[(i, j)] for i in p_indices)
                
                # Constraints for P leaning deviation
                model += p_count - p_plus_dev[j] + p_minus_dev[j] == ideal_p, f"P_Balance_{j}"
                
                # Add to objective with high priority weight
                objective_terms.append(priority_weights['Final_Leaning'] * (p_plus_dev[j] + p_minus_dev[j]))
        
        deviation_vars['Final_Leaning'] = {'plus': p_plus_dev, 'minus': p_minus_dev}
    
    # Process other demographic variables (no changes needed here)
    for var in demographic_vars:
        if var == 'Final_Leaning' or var not in category_counts:
            continue
        
        categories = category_counts[var].keys()
        
        # Create deviation variables for each category
        plus_dev = {cat: LpVariable.dicts(f"{var}_{cat}_Plus_Dev", jury_indices, lowBound=0) 
                   for cat in categories}
        minus_dev = {cat: LpVariable.dicts(f"{var}_{cat}_Minus_Dev", jury_indices, lowBound=0) 
                    for cat in categories}
        
        deviation_vars[var] = {'plus': plus_dev, 'minus': minus_dev}
        
        for cat in categories:
            # Calculate ideal count for this category
            ideal_count = round(category_counts[var][cat] / len(df) * jury_size)
            
            # Get indices of jurors in this category
            if var == 'AgeGroup':
                cat_indices = df[df['AgeGroup'] == cat].index.tolist()
            else:
                cat_indices = df[df[var] == cat].index.tolist()
            
            for j in jury_indices:
                # Count category members in each jury
                cat_count = lpSum(x[(i, j)] for i in cat_indices)
                
                # Constraints for category deviation
                model += cat_count - plus_dev[cat][j] + minus_dev[cat][j] == ideal_count, f"{var}_{cat}_Balance_{j}"
                
                # Add to objective with appropriate priority weight
                objective_terms.append(priority_weights.get(var, 1.0) * (plus_dev[cat][j] + minus_dev[cat][j]))
    
    # STEP 4: Define objective function
    # Minimize the weighted sum of all deviations
    model += lpSum(objective_terms)
    
    # Return the model and important variables
    model_info = {
        'model': model,
        'x': x,
        'deviation_vars': deviation_vars,
        'juror_indices': juror_indices,
        'jury_indices': jury_indices,
        'p_d_balance_info': p_d_balance_info  # Include balance info in result
    }
    
    return model, model_info

def solve_optimization_model(model, model_info, time_limit=300):
    """
    Solve the optimization model.
    
    Parameters:
    model (LpProblem): The optimization model
    model_info (dict): Dictionary containing model variables and information
    time_limit (int, optional): Time limit for solver in seconds
    
    Returns:
    dict: Solution information
    """
    # Set time limit for solver
    solver = PULP_CBC_CMD(timeLimit=time_limit)
    
    # Solve the model
    print("Solving optimization model...")
    model.solve(solver)
    
    # Check solution status
    if LpStatus[model.status] != 'Optimal':
        print(f"Warning: Solution status is {LpStatus[model.status]}")
    
    print(f"Solution status: {LpStatus[model.status]}")
    print(f"Objective value: {value(model.objective)}")
    
    return {
        'status': LpStatus[model.status],
        'objective_value': value(model.objective),
        'model': model,
        'model_info': model_info
    }


def extract_jury_assignments(solution_info, data_dict):
    """
    Extract jury assignments from the solved model.
    
    Parameters:
    solution_info (dict): Solution information from solve_optimization_model
    data_dict (dict): Dictionary containing processed juror data
    
    Returns:
    dict: Jury assignments and analysis
    """
    model = solution_info['model']
    model_info = solution_info['model_info']
    df = data_dict['original_data']
    
    # Print column names in the original dataframe to debug
    print("=== DEBUG: Columns in original dataframe ===")
    print(df.columns.tolist())
    
    x = model_info['x']
    juror_indices = model_info['juror_indices']
    jury_indices = model_info['jury_indices']
    
    # Extract jury assignments
    assignments = []
    for i in juror_indices:
        assigned = False
        for j in jury_indices:
            if value(x[(i, j)]) > 0.5:  # Binary variable is 1 (assigned)
                assignments.append({
                    'juror_index': i,
                    'jury': j
                })
                assigned = True
                break
        if not assigned:
            assignments.append({
                'juror_index': i,
                'jury': 'Unassigned'
            })
    
    # Create DataFrame with assignments
    assignments_df = pd.DataFrame(assignments)
    
    # Merge with original data to get juror information
    juror_data = df.reset_index()
    juror_data = juror_data.rename(columns={'index': 'juror_index'})
    
    jury_assignments = pd.merge(assignments_df, juror_data, on='juror_index')
    
    # Check what columns we have in jury_assignments after merge
    print("=== DEBUG: Columns in jury_assignments ===")
    print(jury_assignments.columns.tolist())
    
    # Analyze jury composition
    jury_analysis = {}
    for j in jury_indices:
        jury_j = jury_assignments[jury_assignments['jury'] == j]
        
        # Start with a basic analysis
        analysis = {
            'size': len(jury_j),
            'jurors': jury_j.to_dict('records')
        }
        
        # Add leaning counts if available
        if 'Final_Leaning' in jury_j.columns:
            analysis['leaning'] = jury_j['Final_Leaning'].value_counts().to_dict()
        
        # Add gender counts if available
        if 'Gender' in jury_j.columns:
            analysis['gender'] = jury_j['Gender'].value_counts().to_dict()
        
        # Add race counts if available
        if 'Race' in jury_j.columns:
            analysis['race'] = jury_j['Race'].value_counts().to_dict()
        
        # Add age group counts if available
        if 'AgeGroup' in jury_j.columns:
            analysis['age_group'] = jury_j['AgeGroup'].value_counts().to_dict()
        
        # Try different possible education column names
        for edu_col in ['Education', 'education', 'EDUCATION', 'Edu', 'edu']:
            if edu_col in jury_j.columns:
                analysis['education'] = jury_j[edu_col].value_counts().to_dict()
                print(f"=== DEBUG: Found education data in column '{edu_col}' ===")
                print(analysis['education'])
                break
        
        # If no education column found but we have data in a custom location, handle that
        
        # Add marital status counts if available
        if 'Marital' in jury_j.columns:
            analysis['marital'] = jury_j['Marital'].value_counts().to_dict()
        
        jury_analysis[j] = analysis
    
     # Add analysis for unassigned jurors
    unassigned_jurors = jury_assignments[jury_assignments['jury'] == 'Unassigned']
    if len(unassigned_jurors) > 0:
        unassigned_analysis = {
            'size': len(unassigned_jurors),
            'jurors': unassigned_jurors.to_dict('records')
        }
        
        # Add demographic breakdowns for unassigned jurors too
        if 'Final_Leaning' in unassigned_jurors.columns:
            unassigned_analysis['leaning'] = unassigned_jurors['Final_Leaning'].value_counts().to_dict()
        
        if 'Gender' in unassigned_jurors.columns:
            unassigned_analysis['gender'] = unassigned_jurors['Gender'].value_counts().to_dict()
        
        if 'Race' in unassigned_jurors.columns:
            unassigned_analysis['race'] = unassigned_jurors['Race'].value_counts().to_dict()
        
        if 'AgeGroup' in unassigned_jurors.columns:
            unassigned_analysis['age_group'] = unassigned_jurors['AgeGroup'].value_counts().to_dict()
            
        # Check for education column
        for edu_col in ['Education', 'education', 'EDUCATION', 'Edu', 'edu']:
            if edu_col in unassigned_jurors.columns:
                unassigned_analysis['education'] = unassigned_jurors[edu_col].value_counts().to_dict()
                break
                
        if 'Marital' in unassigned_jurors.columns:
            unassigned_analysis['marital'] = unassigned_jurors['Marital'].value_counts().to_dict()
            
        jury_analysis['Unassigned'] = unassigned_analysis
    
    # Print the keys for the first jury to verify education was added
    first_jury = next(iter(jury_analysis.values())) if jury_analysis else {}
    print("=== DEBUG: Keys in first jury analysis ===")
    print(list(first_jury.keys()))
    
    # Calculate overall deviation metrics
    deviations = {}
    for var in model_info['deviation_vars']:
        var_dev = 0
        if 'plus' in model_info['deviation_vars'][var] and 'minus' in model_info['deviation_vars'][var]:
            # Handle leaning differently
            if var == 'Final_Leaning':
                plus_dev = model_info['deviation_vars'][var]['plus']
                minus_dev = model_info['deviation_vars'][var]['minus']
                
                for j in jury_indices:
                    var_dev += value(plus_dev[j]) + value(minus_dev[j])
            else:
                plus_dev = model_info['deviation_vars'][var]['plus']
                minus_dev = model_info['deviation_vars'][var]['minus']
                
                for cat in plus_dev:
                    for j in jury_indices:
                        var_dev += value(plus_dev[cat][j]) + value(minus_dev[cat][j])
        
        deviations[var] = var_dev
    
    return {
        'assignments': jury_assignments,
        'jury_analysis': jury_analysis,
        'deviations': deviations,
        'solution_status': solution_info['status'],
        'objective_value': solution_info['objective_value'],
        'unassigned_count': len(unassigned_jurors)  # Add count of unassigned jurors to results
    }

def format_results_for_output(assignment_results, data_dict):
    jury_assignments = assignment_results['assignments']
    jury_analysis = assignment_results['jury_analysis']
    
    # Create a summary DataFrame for each jury
    jury_summaries = []
    
    for jury_num, analysis in jury_analysis.items():
        # Handle both numeric and alphabetic jury IDs
        if isinstance(jury_num, str) and jury_num.isalpha():
            # If it's already a letter like 'A', use it directly
            jury_label = jury_num
        else:
            # Otherwise convert number to letter
            try:
                jury_label = chr(64 + int(jury_num))
            except (ValueError, TypeError):
                # Fallback in case of conversion error
                jury_label = f"Jury {jury_num}"
        
        # Count P and D leanings
        # Check if the 'leaning' key exists, and if not, try 'Final_Leaning'
        leaning_key = None
        if 'leaning' in analysis:
            leaning_key = 'leaning'
        
        summary = {
            'Jury': jury_label,
            'Size': analysis['size'],
        }
        
        # Add P/D leaning counts if available
        if leaning_key:
            p_count = sum(count for leaning, count in analysis[leaning_key].items() 
                        if leaning in ['P', 'P+'])
            d_count = sum(count for leaning, count in analysis[leaning_key].items() 
                        if leaning in ['D', 'D+'])
            
            summary['P_Leaning'] = p_count
            summary['D_Leaning'] = d_count
            summary['P_D_Ratio'] = f"{p_count}:{d_count}"
        else:
            # Set defaults if leaning data is not available
            summary['P_Leaning'] = 0
            summary['D_Leaning'] = 0
            summary['P_D_Ratio'] = "0:0"

        # Add gender distribution if available
        if 'gender' in analysis and analysis['gender']:
            for gender, count in analysis['gender'].items():
                summary[f'Gender_{gender}'] = count
        
        # Add race distribution if available
        if 'race' in analysis and analysis['race']:
            for race, count in analysis['race'].items():
                summary[f'Race_{race}'] = count
        
        # Add age group distribution if available
        if 'age_group' in analysis and analysis['age_group']:
            for age, count in analysis['age_group'].items():
                summary[f'Age_{age}'] = count
        
        # Around line 414 in optimization.py
        if 'education' in analysis and analysis['education']:
            # Handle both numeric and alphabetic jury IDs in debug prints
            if isinstance(jury_num, str) and jury_num.isalpha():
                jury_label = jury_num  # Already a letter like 'A'
            else:
                try:
                    jury_label = chr(64 + int(jury_num))
                except (ValueError, TypeError):
                    jury_label = str(jury_num)
                    
            print(f"Adding education for jury {jury_label}: {analysis['education']}")
            for edu, count in analysis['education'].items():
                summary[f'Education_{edu}'] = count
        else:
            # Same conversion for this debug statement
            if isinstance(jury_num, str) and jury_num.isalpha():
                jury_label = jury_num
            else:
                try:
                    jury_label = chr(64 + int(jury_num))
                except (ValueError, TypeError):
                    jury_label = str(jury_num)
                    
            print(f"No education data found for jury {jury_label}")
            if 'education' in analysis:
                print(f"Education key exists but is empty or not a dict: {analysis['education']}")
        
        jury_summaries.append(summary)
    
    print("Summary data before DataFrame creation:")
    for summary in jury_summaries:
        print(f"Jury {summary['Jury']} summary keys: {list(summary.keys())}")
    
    summary_df = pd.DataFrame(jury_summaries)
    
    print("Summary DataFrame columns:")
    print(summary_df.columns.tolist())
    
    # Format detailed assignments
    detailed_assignments = jury_assignments.sort_values(['jury', 'Name'])
    
    # Select relevant columns
    output_columns = ['jury', 'Name', '#', 'Final_Leaning', 'Gender', 'Race', 'Age', 'AgeGroup', 'Education', 'Marital']
    # Make sure we only include columns that actually exist
    output_columns = [col for col in output_columns if col in detailed_assignments.columns]
    
    detailed_output = detailed_assignments[output_columns].copy()
    # Add alphabetic jury ID column
    detailed_output['jury_letter'] = detailed_output['jury'].apply(lambda x: chr(64 + int(x)) if isinstance(x, (int, float)) else x)
    
    # Either replace the 'jury' column or ensure it's excluded when writing to Excel
    # Option 1: Replace numeric with alphabetic
    detailed_output['jury'] = detailed_output['jury_letter']
    detailed_output = detailed_output.drop('jury_letter', axis=1)

    # Calculate overall balance metrics
    balance_metrics = {}
    
    # Leaning balance
    leaning_col = 'Final_Leaning'
    if leaning_col in detailed_assignments.columns:
        overall_p = detailed_assignments[leaning_col].isin(['P', 'P+']).sum()
        overall_d = detailed_assignments[leaning_col].isin(['D', 'D+']).sum()
        
        leaning_balance = {
            'overall_p': overall_p,
            'overall_d': overall_d,
            'p_percentage': round(overall_p / len(detailed_assignments) * 100, 2),
            'deviation_metric': assignment_results['deviations'].get('Final_Leaning', 0)
        }
        
        balance_metrics['leaning'] = leaning_balance
    
    # Return formatted results
    return {
        'summary': summary_df,
        'detailed_assignments': detailed_output,
        'balance_metrics': balance_metrics,
        'jury_analysis': jury_analysis,
        'solution_quality': {
            'status': assignment_results['solution_status'],
            'objective_value': assignment_results['objective_value'],
            'deviations': assignment_results['deviations']
        }
    }


def optimize_jury_assignment(data_dict, priority_weights=None, time_limit=300):
    """
    Main function to perform jury assignment optimization.
    
    Parameters:
    data_dict (dict): Dictionary containing processed juror data from data_processing
    priority_weights (dict, optional): Dictionary mapping demographic variables to priority weights
    time_limit (int, optional): Time limit for solver in seconds
    
    Returns:
    dict: Optimization results
    """
    # Create the optimization model
    model, model_info = create_optimization_model(data_dict, priority_weights)
    
    # Solve the model
    solution_info = solve_optimization_model(model, model_info, time_limit)
    
    # Extract jury assignments
    assignment_results = extract_jury_assignments(solution_info, data_dict)
    
    # Format results
    formatted_results = format_results_for_output(assignment_results, data_dict)
    
    return formatted_results


if __name__ == "__main__":
    # Example usage (for testing)
    from data_processing import process_juror_data
    
    file_path = r"C:\Users\NicholasWilson\OneDrive - Trial Behavior Consulting\AutoJurySplit_Data.xlsx"
    num_juries = 2
    jury_size = 12
    
    try:
        # Process the data
        data_dict = process_juror_data(file_path, num_juries, jury_size)
        
        # Define priority weights (optional)
        priority_weights = {
            'Final_Leaning': 5.0,
            'Gender': 4.0,
            'Race': 3.0,
            'AgeGroup': 2.0,
            'Education': 1.0,
            'Marital': 0.5
        }
        
        # Run optimization
        results = optimize_jury_assignment(data_dict, priority_weights)
        
        # Print summary
        print("\nJury Assignment Summary:")
        print(results['summary'])
        
        # Print solution quality
        print("\nSolution Quality:")
        print(f"Status: {results['solution_quality']['status']}")
        print(f"Objective Value: {results['solution_quality']['objective_value']}")
        
        # Print balance metrics
        if 'leaning' in results['balance_metrics']:
            leaning = results['balance_metrics']['leaning']
            print(f"\nOverall Leaning Balance: {leaning['overall_p']} P : {leaning['overall_d']} D")
        
    except Exception as e:
        print(f"Error in optimization process: {str(e)}")