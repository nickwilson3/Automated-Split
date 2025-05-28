# optimization.py
import numpy as np
import pandas as pd
from pulp import *

def create_optimization_model(data_dict, priority_weights=None):
    """
    Create the Mixed Integer Programming model for jury assignment optimization.
    Now implements hierarchical optimization: P/D first (hard constraints), 
    then Gender (secondary hard constraints), then other demographics (weighted).
    
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
    
    # Extract enhanced balance info
    balance_info = data_dict.get('balance_info', {})
    
    # Set default priority weights if not provided
    if priority_weights is None:
        # Default priority: Leaning and Gender are hard constraints, others are weighted
        priority_weights = {
            'Final_Leaning': 5.0,  # Hard constraint (handled separately)
            'Gender': 5.0,         # Hard constraint (handled separately)
            'Race': 3.0,
            'AgeGroup': 2.0,
            'Education': 1.0,
            'Marital': 0.5
        }
    
    # Create the optimization model
    model = LpProblem("Jury_Assignment_Optimization", LpMinimize)
    
    # STEP 2: Define decision variables
    # Binary variables x[i,j] = 1 if juror i is assigned to jury j
    juror_indices = df.index.tolist()
    jury_indices = range(1, num_juries + 1)
    
    x = LpVariable.dicts("Assign", 
                         [(i, j) for i in juror_indices for j in jury_indices],
                         cat=LpBinary)
    
    # Define deviation variables for demographics that use weighted optimization
    deviation_vars = {}
    
    # STEP 3: Define constraints
    # Constraint A: Each jury must have exactly jury_size jurors
    for j in jury_indices:
        model += lpSum(x[(i, j)] for i in juror_indices) == jury_size, f"Jury_Size_{j}"
    
    # Constraint B: Each juror must be assigned to exactly one jury (or remain unassigned)
    for i in juror_indices:
        model += lpSum(x[(i, j)] for j in jury_indices) <= 1, f"Juror_Assignment_{i}"
    
    # STEP 4: HIERARCHICAL CONSTRAINTS
    
    # LEVEL 1: P/D Leaning Balance (Hard Constraints - Highest Priority)
    objective_terms = []
    
    if 'Final_Leaning' in demographic_vars and 'Final_Leaning' in df.columns:
        p_indices = df[df['Final_Leaning'].isin(['P', 'P+'])].index.tolist()
        d_indices = df[df['Final_Leaning'].isin(['D', 'D+'])].index.tolist()
        
        # Use the pre-calculated optimal distributions from balance_info
        optimal_p_dist = balance_info.get('optimal_p_distribution')
        optimal_d_dist = balance_info.get('optimal_d_distribution')
        
        if optimal_p_dist and len(optimal_p_dist) == num_juries:
            # We have optimal P distribution - enforce as hard constraints
            for j in jury_indices:
                target_p = optimal_p_dist[j-1]  # j is 1-indexed, list is 0-indexed
                p_count = lpSum(x[(i, j)] for i in p_indices)
                model += p_count == target_p, f"P_Balance_Hard_{j}"
                print(f"Added hard constraint: Jury {j} must have exactly {target_p} P jurors")
        
        elif optimal_d_dist and len(optimal_d_dist) == num_juries:
            # We have optimal D distribution - enforce as hard constraints
            for j in jury_indices:
                target_d = optimal_d_dist[j-1]  # j is 1-indexed, list is 0-indexed
                d_count = lpSum(x[(i, j)] for i in d_indices)
                model += d_count == target_d, f"D_Balance_Hard_{j}"
                print(f"Added hard constraint: Jury {j} must have exactly {target_d} D jurors")
        
        else:
            # Fallback: use deviation variables for P/D balance (shouldn't happen with good data processing)
            print("Warning: Using fallback P/D deviation constraints")
            p_plus_dev = LpVariable.dicts("P_Plus_Dev", jury_indices, lowBound=0)
            p_minus_dev = LpVariable.dicts("P_Minus_Dev", jury_indices, lowBound=0)
            
            ideal_p = jury_size // 2
            for j in jury_indices:
                p_count = lpSum(x[(i, j)] for i in p_indices)
                model += p_count - p_plus_dev[j] + p_minus_dev[j] == ideal_p, f"P_Balance_Soft_{j}"
                objective_terms.append(1000.0 * (p_plus_dev[j] + p_minus_dev[j]))  # Very high weight
            
            deviation_vars['Final_Leaning'] = {'plus': p_plus_dev, 'minus': p_minus_dev}
    
    # LEVEL 2: Gender Balance (Secondary Hard Constraints)
    
    if 'Gender' in demographic_vars and 'Gender' in df.columns:
        # Get male and female juror indices
        male_indices = df[df['Gender'].isin(['M', 'Male', 'male', 'MALE'])].index.tolist()
        female_indices = df[df['Gender'].isin(['F', 'Female', 'female', 'FEMALE'])].index.tolist()
        
        # Use the pre-calculated optimal gender distributions from balance_info
        optimal_male_dist = balance_info.get('optimal_male_distribution')
        optimal_female_dist = balance_info.get('optimal_female_distribution')
        
        if optimal_male_dist and len(optimal_male_dist) == num_juries:
            # We have optimal male distribution - enforce as hard constraints
            for j in jury_indices:
                target_male = optimal_male_dist[j-1]  # j is 1-indexed, list is 0-indexed
                male_count = lpSum(x[(i, j)] for i in male_indices)
                model += male_count == target_male, f"Male_Balance_Hard_{j}"
                print(f"Added hard constraint: Jury {j} must have exactly {target_male} male jurors")
        
        elif optimal_female_dist and len(optimal_female_dist) == num_juries:
            # We have optimal female distribution - enforce as hard constraints
            for j in jury_indices:
                target_female = optimal_female_dist[j-1]  # j is 1-indexed, list is 0-indexed
                female_count = lpSum(x[(i, j)] for i in female_indices)
                model += female_count == target_female, f"Female_Balance_Hard_{j}"
                print(f"Added hard constraint: Jury {j} must have exactly {target_female} female jurors")
        
        else:
            # Fallback: try to balance gender with deviation variables (minimize imbalance)
            print("Using gender balance deviation constraints")
            gender_plus_dev = LpVariable.dicts("Gender_Plus_Dev", jury_indices, lowBound=0)
            gender_minus_dev = LpVariable.dicts("Gender_Minus_Dev", jury_indices, lowBound=0)
            
            ideal_male = jury_size // 2
            for j in jury_indices:
                male_count = lpSum(x[(i, j)] for i in male_indices)
                model += male_count - gender_plus_dev[j] + gender_minus_dev[j] == ideal_male, f"Gender_Balance_Soft_{j}"
                # High weight but lower than P/D leaning
                objective_terms.append(100.0 * (gender_plus_dev[j] + gender_minus_dev[j]))
            
            deviation_vars['Gender'] = {'plus': gender_plus_dev, 'minus': gender_minus_dev}
    
    # LEVEL 3: Other Demographics (Weighted Optimization)
    
    # Process remaining demographic variables with weighted optimization
    remaining_vars = [var for var in demographic_vars if var not in ['Final_Leaning', 'Gender']]
    
    for var in remaining_vars:
        if var not in category_counts:
            continue
        
        print(f"Adding weighted constraints for {var}")
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
                
                # Add to objective with appropriate priority weight (much lower than hard constraints)
                weight = priority_weights.get(var, 1.0)
                objective_terms.append(weight * (plus_dev[cat][j] + minus_dev[cat][j]))
    
    # STEP 5: Define objective function
    # Minimize the weighted sum of all deviations (only for non-hard-constraint demographics)
    if objective_terms:
        model += lpSum(objective_terms)
    else:
        # If no objective terms (all demographics have hard constraints), just minimize a dummy variable
        dummy_var = LpVariable("Dummy_Objective", lowBound=0)
        model += dummy_var
        model += dummy_var == 0  # Force dummy to be 0
    
    # Print constraint summary
    print(f"Optimization model created with {len(model.constraints)} constraints")
    print("Constraint types:")
    print(f"  - Jury size constraints: {num_juries}")
    print(f"  - Juror assignment constraints: {len(juror_indices)}")
    print(f"  - P/D hard constraints: {num_juries if balance_info.get('optimal_p_distribution') or balance_info.get('optimal_d_distribution') else 0}")
    print(f"  - Gender hard constraints: {num_juries if balance_info.get('optimal_male_distribution') or balance_info.get('optimal_female_distribution') else 0}")
    print(f"  - Weighted demographic constraints: {len(remaining_vars) * sum(len(category_counts.get(var, {})) for var in remaining_vars) * num_juries}")
    
    # Return the model and important variables
    model_info = {
        'model': model,
        'x': x,
        'deviation_vars': deviation_vars,
        'juror_indices': juror_indices,
        'jury_indices': jury_indices,
        'balance_info': balance_info,
        'has_hard_constraints': {
            'leaning': bool(balance_info.get('optimal_p_distribution') or balance_info.get('optimal_d_distribution')),
            'gender': bool(balance_info.get('optimal_male_distribution') or balance_info.get('optimal_female_distribution'))
        }
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
    print(f"Model has {len(model.variables())} variables and {len(model.constraints)} constraints")
    
    # Print information about hard constraints
    has_hard = model_info.get('has_hard_constraints', {})
    if has_hard.get('leaning'):
        print("Using HARD constraints for P/D leaning balance")
    if has_hard.get('gender'):
        print("Using HARD constraints for gender balance")
    
    model.solve(solver)
    
    # Check solution status
    status = LpStatus[model.status]
    print(f"Solution status: {status}")
    
    if status == 'Optimal':
        print(f"Objective value: {value(model.objective)}")
    elif status == 'Infeasible':
        print("ERROR: Model is infeasible - constraints cannot be satisfied")
        print("This may indicate:")
        print("  - Not enough jurors of required demographics")
        print("  - Conflicting hard constraints for P/D and gender balance")
        print("  - Data processing error")
    elif status in ['Not Solved', 'Undefined']:
        print(f"WARNING: Solver did not find optimal solution: {status}")
    
    return {
        'status': status,
        'objective_value': value(model.objective) if status == 'Optimal' else None,
        'model': model,
        'model_info': model_info
    }


def extract_jury_assignments(solution_info, data_dict):
    """
    Extract jury assignments from the solved model.
    Enhanced to handle hierarchical optimization results.
    
    Parameters:
    solution_info (dict): Solution information from solve_optimization_model
    data_dict (dict): Dictionary containing processed juror data
    
    Returns:
    dict: Jury assignments and analysis
    """
    model = solution_info['model']
    model_info = solution_info['model_info']
    df = data_dict['original_data']
    balance_info = data_dict.get('balance_info', {})
    
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
    
    # Analyze jury composition with enhanced balance checking
    jury_analysis = {}
    balance_achieved = {
        'leaning': True,
        'gender': True,
        'details': {}
    }
    
    for j in jury_indices:
        jury_j = jury_assignments[jury_assignments['jury'] == j]
        
        # Start with basic analysis
        analysis = {
            'size': len(jury_j),
            'jurors': jury_j.to_dict('records')
        }
        
        # Add leaning analysis with balance checking
        if 'Final_Leaning' in jury_j.columns:
            leaning_counts = jury_j['Final_Leaning'].value_counts().to_dict()
            analysis['leaning'] = leaning_counts
            
            # Check if this jury achieved optimal P/D balance
            p_count = sum(count for leaning, count in leaning_counts.items() if leaning in ['P', 'P+'])
            d_count = sum(count for leaning, count in leaning_counts.items() if leaning in ['D', 'D+'])
            
            optimal_p_dist = balance_info.get('optimal_p_distribution') or []
            optimal_d_dist = balance_info.get('optimal_d_distribution') or []
            
            target_p = optimal_p_dist[j-1] if optimal_p_dist and j-1 < len(optimal_p_dist) else None
            target_d = optimal_d_dist[j-1] if optimal_d_dist and j-1 < len(optimal_d_dist) else None
            
            jury_leaning_optimal = True
            if target_p is not None and p_count != target_p:
                jury_leaning_optimal = False
                balance_achieved['leaning'] = False
            if target_d is not None and d_count != target_d:
                jury_leaning_optimal = False
                balance_achieved['leaning'] = False
            
            balance_achieved['details'][f'jury_{j}_leaning_optimal'] = jury_leaning_optimal
        
        # Add gender analysis with balance checking
        if 'Gender' in jury_j.columns:
            gender_counts = jury_j['Gender'].value_counts().to_dict()
            analysis['gender'] = gender_counts
            
            # Check if this jury achieved optimal gender balance
            male_count = sum(count for gender, count in gender_counts.items() if gender in ['M', 'Male', 'male', 'MALE'])
            female_count = sum(count for gender, count in gender_counts.items() if gender in ['F', 'Female', 'female', 'FEMALE'])
            
            optimal_male_dist = balance_info.get('optimal_male_distribution') or []
            optimal_female_dist = balance_info.get('optimal_female_distribution') or []
            
            target_male = optimal_male_dist[j-1] if optimal_male_dist and j-1 < len(optimal_male_dist) else None
            target_female = optimal_female_dist[j-1] if optimal_female_dist and j-1 < len(optimal_female_dist) else None
            
            jury_gender_optimal = True
            if target_male is not None and male_count != target_male:
                jury_gender_optimal = False
                balance_achieved['gender'] = False
            if target_female is not None and female_count != target_female:
                jury_gender_optimal = False
                balance_achieved['gender'] = False
            
            balance_achieved['details'][f'jury_{j}_gender_optimal'] = jury_gender_optimal
        
        # Add other demographic analyses
        if 'Race' in jury_j.columns:
            analysis['race'] = jury_j['Race'].value_counts().to_dict()
        
        if 'AgeGroup' in jury_j.columns:
            analysis['age_group'] = jury_j['AgeGroup'].value_counts().to_dict()
        
        # Try different possible education column names
        for edu_col in ['Education', 'education', 'EDUCATION', 'Edu', 'edu']:
            if edu_col in jury_j.columns:
                analysis['education'] = jury_j[edu_col].value_counts().to_dict()
                print(f"=== DEBUG: Found education data in column '{edu_col}' ===")
                print(analysis['education'])
                break
        
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
    
    # Calculate overall deviation metrics (only for weighted demographics now)
    deviations = {}
    for var in model_info['deviation_vars']:
        var_dev = 0
        if 'plus' in model_info['deviation_vars'][var] and 'minus' in model_info['deviation_vars'][var]:
            # Handle leaning and gender differently based on whether they used hard constraints
            if var in ['Final_Leaning', 'Gender']:
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
    
    # Print balance achievement summary
    print("=== BALANCE ACHIEVEMENT SUMMARY ===")
    print(f"Optimal P/D leaning balance achieved: {balance_achieved['leaning']}")
    print(f"Optimal gender balance achieved: {balance_achieved['gender']}")
    
    return {
        'assignments': jury_assignments,
        'jury_analysis': jury_analysis,
        'deviations': deviations,
        'solution_status': solution_info['status'],
        'objective_value': solution_info['objective_value'],
        'unassigned_count': len(unassigned_jurors),
        'balance_achieved': balance_achieved,  # New field to track hierarchical balance success
        'used_hard_constraints': model_info.get('has_hard_constraints', {})
    }

def format_results_for_output(assignment_results, data_dict):
    """
    Format the assignment results for output with enhanced balance reporting.
    """
    jury_assignments = assignment_results['assignments']
    jury_analysis = assignment_results['jury_analysis']
    
    # Create a summary DataFrame for each jury
    jury_summaries = []
    
    for jury_num, analysis in jury_analysis.items():
        # Handle both numeric and alphabetic jury IDs
        if isinstance(jury_num, str) and jury_num.isalpha():
            jury_label = jury_num
        else:
            try:
                jury_label = chr(64 + int(jury_num))
            except (ValueError, TypeError):
                jury_label = f"Jury {jury_num}"
        
        # Count P and D leanings
        leaning_key = 'leaning' if 'leaning' in analysis else None
        
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
        
        # Add education distribution
        if 'education' in analysis and analysis['education']:
            if isinstance(jury_num, str) and jury_num.isalpha():
                jury_label_debug = jury_num
            else:
                try:
                    jury_label_debug = chr(64 + int(jury_num))
                except (ValueError, TypeError):
                    jury_label_debug = str(jury_num)
                    
            print(f"Adding education for jury {jury_label_debug}: {analysis['education']}")
            for edu, count in analysis['education'].items():
                summary[f'Education_{edu}'] = count
        else:
            if isinstance(jury_num, str) and jury_num.isalpha():
                jury_label_debug = jury_num
            else:
                try:
                    jury_label_debug = chr(64 + int(jury_num))
                except (ValueError, TypeError):
                    jury_label_debug = str(jury_num)
                    
            print(f"No education data found for jury {jury_label_debug}")
        
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
    output_columns = [col for col in output_columns if col in detailed_assignments.columns]
    
    detailed_output = detailed_assignments[output_columns].copy()
    # Add alphabetic jury ID column
    detailed_output['jury_letter'] = detailed_output['jury'].apply(lambda x: chr(64 + int(x)) if isinstance(x, (int, float)) else x)
    
    # Replace numeric with alphabetic
    detailed_output['jury'] = detailed_output['jury_letter']
    detailed_output = detailed_output.drop('jury_letter', axis=1)

    # Calculate overall balance metrics with hierarchical info
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
            'deviation_metric': assignment_results['deviations'].get('Final_Leaning', 0),
            'optimal_achieved': assignment_results.get('balance_achieved', {}).get('leaning', False),
            'used_hard_constraints': assignment_results.get('used_hard_constraints', {}).get('leaning', False)
        }
        
        balance_metrics['leaning'] = leaning_balance
    
    # Gender balance
    if 'Gender' in detailed_assignments.columns:
        overall_male = detailed_assignments['Gender'].isin(['M', 'Male', 'male', 'MALE']).sum()
        overall_female = detailed_assignments['Gender'].isin(['F', 'Female', 'female', 'FEMALE']).sum()
        
        gender_balance = {
            'overall_male': overall_male,
            'overall_female': overall_female,
            'male_percentage': round(overall_male / len(detailed_assignments) * 100, 2),
            'deviation_metric': assignment_results['deviations'].get('Gender', 0),
            'optimal_achieved': assignment_results.get('balance_achieved', {}).get('gender', False),
            'used_hard_constraints': assignment_results.get('used_hard_constraints', {}).get('gender', False)
        }
        
        balance_metrics['gender'] = gender_balance
    
    # Return formatted results with enhanced balance information
    return {
        'summary': summary_df,
        'detailed_assignments': detailed_output,
        'balance_metrics': balance_metrics,
        'jury_analysis': jury_analysis,
        'solution_quality': {
            'status': assignment_results['solution_status'],
            'objective_value': assignment_results['objective_value'],
            'deviations': assignment_results['deviations'],
            'hierarchical_balance': assignment_results.get('balance_achieved', {}),
            'constraint_types_used': assignment_results.get('used_hard_constraints', {})
        }
    }


def optimize_jury_assignment(data_dict, priority_weights=None, time_limit=300):
    """
    Main function to perform jury assignment optimization with hierarchical constraints.
    
    Parameters:
    data_dict (dict): Dictionary containing processed juror data from data_processing
    priority_weights (dict, optional): Dictionary mapping demographic variables to priority weights
    time_limit (int, optional): Time limit for solver in seconds
    
    Returns:
    dict: Optimization results with hierarchical balance information
    """
    print("Starting hierarchical jury optimization...")
    print("Priority hierarchy: 1) P/D Leaning (hard), 2) Gender (hard), 3) Other demographics (weighted)")
    
    # Create the optimization model with hierarchical constraints
    model, model_info = create_optimization_model(data_dict, priority_weights)
    
    # Solve the model
    solution_info = solve_optimization_model(model, model_info, time_limit)
    
    # Check if solution was successful
    if solution_info['status'] != 'Optimal':
        print(f"WARNING: Optimization did not find optimal solution: {solution_info['status']}")
        if solution_info['status'] == 'Infeasible':
            print("Consider relaxing constraints or checking data quality")
    
    # Extract jury assignments
    assignment_results = extract_jury_assignments(solution_info, data_dict)
    
    # Format results
    formatted_results = format_results_for_output(assignment_results, data_dict)
    
    # Print final summary
    print("=== OPTIMIZATION COMPLETE ===")
    if 'hierarchical_balance' in formatted_results['solution_quality']:
        balance_info = formatted_results['solution_quality']['hierarchical_balance']
        print(f"P/D Balance achieved: {balance_info.get('leaning', 'Unknown')}")
        print(f"Gender Balance achieved: {balance_info.get('gender', 'Unknown')}")
    
    constraint_types = formatted_results['solution_quality'].get('constraint_types_used', {})
    print(f"Used hard constraints for P/D: {constraint_types.get('leaning', False)}")
    print(f"Used hard constraints for Gender: {constraint_types.get('gender', False)}")
    
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
        
        # Define priority weights (for non-hard-constraint demographics)
        priority_weights = {
            'Final_Leaning': 5.0,  # Will use hard constraints
            'Gender': 5.0,         # Will use hard constraints  
            'Race': 3.0,           # Weighted optimization
            'AgeGroup': 2.0,       # Weighted optimization
            'Education': 1.0,      # Weighted optimization
            'Marital': 0.5         # Weighted optimization
        }
        
        # Run hierarchical optimization
        results = optimize_jury_assignment(data_dict, priority_weights)
        
        # Print summary
        print("\nJury Assignment Summary:")
        print(results['summary'])
        
        # Print solution quality
        print("\nSolution Quality:")
        solution_quality = results['solution_quality']
        print(f"Status: {solution_quality['status']}")
        print(f"Objective Value: {solution_quality['objective_value']}")
        
        # Print hierarchical balance results
        if 'hierarchical_balance' in solution_quality:
            print(f"P/D Balance Optimal: {solution_quality['hierarchical_balance'].get('leaning', 'Unknown')}")
            print(f"Gender Balance Optimal: {solution_quality['hierarchical_balance'].get('gender', 'Unknown')}")
        
        # Print balance metrics
        if 'leaning' in results['balance_metrics']:
            leaning = results['balance_metrics']['leaning']
            print(f"\nOverall Leaning Balance: {leaning['overall_p']} P : {leaning['overall_d']} D")
            print(f"Used hard constraints for leaning: {leaning.get('used_hard_constraints', False)}")
            
        if 'gender' in results['balance_metrics']:
            gender = results['balance_metrics']['gender']
            print(f"Overall Gender Balance: {gender['overall_male']} M : {gender['overall_female']} F")
            print(f"Used hard constraints for gender: {gender.get('used_hard_constraints', False)}")
        
    except Exception as e:
        import traceback
        print(f"Error in optimization process: {str(e)}")
        print(traceback.format_exc())