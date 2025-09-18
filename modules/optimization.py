# optimization.py
import numpy as np
import pandas as pd
from pulp import *

def create_single_jury_optimization_model(data_dict, jury_number, remaining_juror_indices, target_distributions, priority_weights=None):
    """
    Create optimization model for a single jury using sequential approach.
    
    Parameters:
    data_dict (dict): Dictionary containing processed juror data and parameters
    jury_number (int): Which jury we're optimizing (1, 2, 3, etc.)
    remaining_juror_indices (list): Indices of jurors still available for assignment
    target_distributions (dict): Target distributions for this specific jury
    priority_weights (dict, optional): Priority weights for tertiary demographics
    
    Returns:
    tuple: (model, model_info)
    """
    # Extract data from the dictionary
    df = data_dict['original_data']
    jury_size = data_dict['jury_size']
    category_counts = data_dict['category_counts']
    demographic_vars = data_dict['demographic_vars']
    
    # Set default priority weights (only affects tertiary level)
    if priority_weights is None:
        priority_weights = {
            'Race': 3.0,
            'AgeGroup': 2.0,
            'Education': 1.0,
            'Marital': 0.5
        }
    
    # Create the optimization model for single jury
    model = LpProblem(f"Sequential_Jury_{jury_number}_Optimization", LpMinimize)
    
    # Define decision variables for this jury only
    # Binary variables x[i] = 1 if juror i is assigned to this jury
    x = LpVariable.dicts(f"Assign_Jury_{jury_number}", remaining_juror_indices, cat=LpBinary)
    
    # Define deviation variables for soft constraints
    deviation_vars = {}
    objective_terms = []
    
    # ========== TIER 1: INVIOLABLE HARD CONSTRAINTS ==========
    
    print(f"Adding TIER 1 constraints for Jury {jury_number}...")
    
    # Constraint 1: This jury must have exactly jury_size jurors
    model += lpSum(x[i] for i in remaining_juror_indices) == jury_size, f"Jury_{jury_number}_Size_INVIOLABLE"
    print(f"TIER 1: Jury {jury_number} must have exactly {jury_size} jurors")
    
    # Constraint 2: Basic P/D balance for this jury
    if 'Final_Leaning' in demographic_vars and 'Final_Leaning' in df.columns:
        # Get overall P and D indices from remaining jurors
        remaining_df = df.loc[remaining_juror_indices]
        p_overall_indices = remaining_df[remaining_df['Final_Leaning'].isin(['P', 'P+'])].index.tolist()
        d_overall_indices = remaining_df[remaining_df['Final_Leaning'].isin(['D', 'D+'])].index.tolist()
        
        # Use target distributions for this jury
        target_p_overall = target_distributions.get('p_overall_target', jury_size // 2)
        target_d_overall = target_distributions.get('d_overall_target', jury_size // 2)
        
        # Hard constraints for basic P/D balance
        if p_overall_indices:
            p_overall_count = lpSum(x[i] for i in p_overall_indices if i in remaining_juror_indices)
            model += p_overall_count == target_p_overall, f"P_Overall_INVIOLABLE_Jury_{jury_number}"
            print(f"TIER 1: Jury {jury_number} must have exactly {target_p_overall} overall P jurors")
        
        if d_overall_indices:
            d_overall_count = lpSum(x[i] for i in d_overall_indices if i in remaining_juror_indices)
            model += d_overall_count == target_d_overall, f"D_Overall_INVIOLABLE_Jury_{jury_number}"
            print(f"TIER 1: Jury {jury_number} must have exactly {target_d_overall} overall D jurors")
    
    # ========== TIER 2: SECONDARY SOFT CONSTRAINTS (HIGH PENALTY) ==========
    
    print(f"Adding TIER 2 constraints for Jury {jury_number}...")
    
    # Granular P+/P/D/D+ balance
    if 'Final_Leaning' in demographic_vars and 'Final_Leaning' in df.columns:
        remaining_df = df.loc[remaining_juror_indices]
        
        # Get indices for each granular leaning category from remaining jurors
        granular_indices = {
            'P+': remaining_df[remaining_df['Final_Leaning'] == 'P+'].index.tolist(),
            'P': remaining_df[remaining_df['Final_Leaning'] == 'P'].index.tolist(),
            'D': remaining_df[remaining_df['Final_Leaning'] == 'D'].index.tolist(),
            'D+': remaining_df[remaining_df['Final_Leaning'] == 'D+'].index.tolist()
        }
        
        # Use target granular distributions for this jury
        granular_targets = target_distributions.get('granular_targets', {})
        
        for category in ['P+', 'P', 'D', 'D+']:
            indices = granular_indices[category]
            target_count = granular_targets.get(category, 0)
            
            if indices and target_count > 0:
                # Create deviation variables for granular balance
                plus_dev = LpVariable(f"{category}_Plus_Dev_Jury_{jury_number}", lowBound=0)
                minus_dev = LpVariable(f"{category}_Minus_Dev_Jury_{jury_number}", lowBound=0)
                
                category_count = lpSum(x[i] for i in indices if i in remaining_juror_indices)
                model += category_count - plus_dev + minus_dev == target_count, f"{category}_Granular_Soft_Jury_{jury_number}"
                
                # High penalty for granular deviations (TIER 2)
                objective_terms.append(100.0 * (plus_dev + minus_dev))
                print(f"TIER 2: Jury {jury_number} targets {target_count} {category} jurors (soft constraint)")
                
                deviation_vars[f'{category}_Granular'] = {'plus': plus_dev, 'minus': minus_dev}
    
    # Gender balance (TIER 2)
    if 'Gender' in demographic_vars and 'Gender' in df.columns:
        remaining_df = df.loc[remaining_juror_indices]
        male_indices = remaining_df[remaining_df['Gender'].isin(['M', 'Male', 'male', 'MALE'])].index.tolist()
        female_indices = remaining_df[remaining_df['Gender'].isin(['F', 'Female', 'female', 'FEMALE'])].index.tolist()
        
        # Use target gender distribution for this jury
        target_male = target_distributions.get('male_target', jury_size // 2)
        target_female = target_distributions.get('female_target', jury_size // 2)
        
        if male_indices:
            male_plus_dev = LpVariable(f"Male_Plus_Dev_Jury_{jury_number}", lowBound=0)
            male_minus_dev = LpVariable(f"Male_Minus_Dev_Jury_{jury_number}", lowBound=0)
            
            male_count = lpSum(x[i] for i in male_indices if i in remaining_juror_indices)
            model += male_count - male_plus_dev + male_minus_dev == target_male, f"Male_Balance_Soft_Jury_{jury_number}"
            
            # High penalty for gender deviations (TIER 2)
            objective_terms.append(90.0 * (male_plus_dev + male_minus_dev))
            print(f"TIER 2: Jury {jury_number} targets {target_male} male jurors (soft constraint)")
            
            deviation_vars['Male'] = {'plus': male_plus_dev, 'minus': male_minus_dev}
    
    # ========== TIER 3: TERTIARY WEIGHTED OPTIMIZATION ==========
    
    print(f"Adding TIER 3 constraints for Jury {jury_number}...")
    
    # Process remaining demographic variables with lower priority weights
    remaining_vars = [var for var in demographic_vars if var not in ['Final_Leaning', 'Gender']]
    
    for var in remaining_vars:
        if var not in category_counts:
            continue
        
        print(f"TIER 3: Adding weighted constraints for {var}")
        remaining_df = df.loc[remaining_juror_indices]
        
        # Get categories present in remaining jurors
        if var == 'AgeGroup':
            var_categories = remaining_df['AgeGroup'].dropna().unique() if 'AgeGroup' in remaining_df.columns else []
        else:
            var_categories = remaining_df[var].dropna().unique() if var in remaining_df.columns else []
        
        for cat in var_categories:
            # Calculate ideal count for this category based on remaining jurors
            if var == 'AgeGroup':
                cat_indices = remaining_df[remaining_df['AgeGroup'] == cat].index.tolist()
            else:
                cat_indices = remaining_df[remaining_df[var] == cat].index.tolist()
            
            if not cat_indices:
                continue
            
            # Calculate proportional target
            total_remaining = len(remaining_juror_indices)
            category_remaining = len(cat_indices)
            ideal_count = round((category_remaining / total_remaining) * jury_size) if total_remaining > 0 else 0
            
            if ideal_count > 0:
                # Create deviation variables
                plus_dev = LpVariable(f"{var}_{cat}_Plus_Dev_Jury_{jury_number}", lowBound=0)
                minus_dev = LpVariable(f"{var}_{cat}_Minus_Dev_Jury_{jury_number}", lowBound=0)
                
                cat_count = lpSum(x[i] for i in cat_indices if i in remaining_juror_indices)
                model += cat_count - plus_dev + minus_dev == ideal_count, f"{var}_{cat}_Balance_Jury_{jury_number}"
                
                # Lower priority weight (TIER 3)
                weight = priority_weights.get(var, 1.0)
                objective_terms.append(weight * (plus_dev + minus_dev))
                
                if var not in deviation_vars:
                    deviation_vars[var] = {'plus': {}, 'minus': {}}
                deviation_vars[var]['plus'][cat] = plus_dev
                deviation_vars[var]['minus'][cat] = minus_dev
    
    # ========== OBJECTIVE FUNCTION ==========
    if objective_terms:
        model += lpSum(objective_terms)
        print(f"Jury {jury_number} objective function includes {len(objective_terms)} penalty terms")
    else:
        # If no objective terms, minimize a dummy variable
        dummy_var = LpVariable(f"Dummy_Objective_Jury_{jury_number}", lowBound=0)
        model += dummy_var
        model += dummy_var == 0
        print(f"Jury {jury_number} using dummy objective (no soft constraints)")
    
    # Return the model and important variables
    model_info = {
        'model': model,
        'x': x,
        'deviation_vars': deviation_vars,
        'remaining_juror_indices': remaining_juror_indices,
        'jury_number': jury_number,
        'target_distributions': target_distributions
    }
    
    return model, model_info


def solve_single_jury_optimization(model, model_info, time_limit=300):
    """
    Solve optimization model for a single jury.
    
    Parameters:
    model (LpProblem): The optimization model
    model_info (dict): Dictionary containing model variables and information
    time_limit (int, optional): Time limit for solver in seconds
    
    Returns:
    dict: Solution information
    """
    jury_number = model_info['jury_number']
    
    # Set time limit for solver
    solver = PULP_CBC_CMD(timeLimit=time_limit)
    
    # Solve the model
    print(f"Solving optimization for Jury {jury_number}...")
    model.solve(solver)
    
    # Check solution status
    status = LpStatus[model.status]
    print(f"Jury {jury_number} solution status: {status}")
    
    if status == 'Optimal':
        print(f"Jury {jury_number} objective value: {value(model.objective)}")
    elif status == 'Infeasible':
        print(f"ERROR: Jury {jury_number} optimization is infeasible")
        print("This indicates TIER 1 constraints cannot be satisfied with remaining jurors")
    elif status in ['Not Solved', 'Undefined']:
        print(f"WARNING: Jury {jury_number} solver did not find optimal solution: {status}")
    
    return {
        'status': status,
        'objective_value': value(model.objective) if status == 'Optimal' else None,
        'model': model,
        'model_info': model_info
    }


def extract_single_jury_assignments(solution_info, data_dict):
    """
    Extract jury assignments from a single jury optimization.
    
    Parameters:
    solution_info (dict): Solution information from solve_single_jury_optimization
    data_dict (dict): Dictionary containing processed juror data
    
    Returns:
    dict: Assignment results for this jury
    """
    model_info = solution_info['model_info']
    df = data_dict['original_data']
    
    x = model_info['x']
    remaining_juror_indices = model_info['remaining_juror_indices']
    jury_number = model_info['jury_number']
    
    # Extract assignments for this jury
    assigned_indices = []
    unassigned_indices = []
    
    for i in remaining_juror_indices:
        if value(x[i]) > 0.5:  # Binary variable is 1 (assigned)
            assigned_indices.append(i)
        else:
            unassigned_indices.append(i)
    
    print(f"Jury {jury_number}: Assigned {len(assigned_indices)} jurors, {len(unassigned_indices)} remain unassigned")
    
    return {
        'assigned_indices': assigned_indices,
        'unassigned_indices': unassigned_indices,
        'jury_number': jury_number,
        'solution_status': solution_info['status'],
        'objective_value': solution_info['objective_value']
    }


def calculate_target_distributions_for_jury(remaining_juror_indices, jury_size, data_dict, jury_number, total_juries):
    """
    Calculate target distributions for a specific jury based on remaining jurors.
    Uses nested maximin approach adapted for sequential optimization.
    
    Parameters:
    remaining_juror_indices (list): Indices of jurors still available
    jury_size (int): Size of this jury
    data_dict (dict): Dictionary containing processed juror data
    jury_number (int): Which jury we're optimizing
    total_juries (int): Total number of juries
    
    Returns:
    dict: Target distributions for this jury
    """
    df = data_dict['original_data']
    remaining_df = df.loc[remaining_juror_indices]
    
    print(f"Calculating targets for Jury {jury_number} with {len(remaining_juror_indices)} remaining jurors")
    
    target_distributions = {}
    
    # Calculate overall P/D targets
    if 'Final_Leaning' in remaining_df.columns:
        p_overall_count = remaining_df['Final_Leaning'].isin(['P', 'P+']).sum()
        d_overall_count = remaining_df['Final_Leaning'].isin(['D', 'D+']).sum()
        
        # Granular counts
        granular_counts = {
            'P+': (remaining_df['Final_Leaning'] == 'P+').sum(),
            'P': (remaining_df['Final_Leaning'] == 'P').sum(),
            'D': (remaining_df['Final_Leaning'] == 'D').sum(),
            'D+': (remaining_df['Final_Leaning'] == 'D+').sum()
        }
        
        # Calculate targets for basic P/D balance
        if jury_size % 2 == 0:
            # Even jury size - try for 50/50 split
            ideal_p = jury_size // 2
            ideal_d = jury_size // 2
        else:
            # Odd jury size - give extra to side with more jurors available
            if p_overall_count >= d_overall_count:
                ideal_p = (jury_size + 1) // 2
                ideal_d = jury_size // 2
            else:
                ideal_p = jury_size // 2
                ideal_d = (jury_size + 1) // 2
        
        # Adjust if we don't have enough of either side
        if p_overall_count < ideal_p:
            actual_p = p_overall_count
            actual_d = jury_size - actual_p
        elif d_overall_count < ideal_d:
            actual_d = d_overall_count
            actual_p = jury_size - actual_d
        else:
            actual_p = ideal_p
            actual_d = ideal_d
        
        target_distributions['p_overall_target'] = actual_p
        target_distributions['d_overall_target'] = actual_d
        
        print(f"Jury {jury_number} P/D targets: {actual_p}P, {actual_d}D")
        
        # Calculate granular targets within P/D allocations
        granular_targets = {}
        
        # P-side granular distribution
        if actual_p > 0:
            if actual_p % 2 == 0:
                # Even P allocation - try for equal P+ and P
                target_p_plus = actual_p // 2
                target_p = actual_p // 2
            else:
                # Odd P allocation - give extra to side with more availability
                if granular_counts['P+'] >= granular_counts['P']:
                    target_p_plus = (actual_p + 1) // 2
                    target_p = actual_p // 2
                else:
                    target_p_plus = actual_p // 2
                    target_p = (actual_p + 1) // 2
            
            # Adjust for availability
            if granular_counts['P+'] < target_p_plus:
                target_p_plus = granular_counts['P+']
                target_p = actual_p - target_p_plus
            elif granular_counts['P'] < target_p:
                target_p = granular_counts['P']
                target_p_plus = actual_p - target_p
            
            granular_targets['P+'] = target_p_plus
            granular_targets['P'] = target_p
        else:
            granular_targets['P+'] = 0
            granular_targets['P'] = 0
        
        # D-side granular distribution
        if actual_d > 0:
            if actual_d % 2 == 0:
                # Even D allocation - try for equal D and D+
                target_d = actual_d // 2
                target_d_plus = actual_d // 2
            else:
                # Odd D allocation - give extra to side with more availability
                if granular_counts['D'] >= granular_counts['D+']:
                    target_d = (actual_d + 1) // 2
                    target_d_plus = actual_d // 2
                else:
                    target_d = actual_d // 2
                    target_d_plus = (actual_d + 1) // 2
            
            # Adjust for availability
            if granular_counts['D'] < target_d:
                target_d = granular_counts['D']
                target_d_plus = actual_d - target_d
            elif granular_counts['D+'] < target_d_plus:
                target_d_plus = granular_counts['D+']
                target_d = actual_d - target_d_plus
            
            granular_targets['D'] = target_d
            granular_targets['D+'] = target_d_plus
        else:
            granular_targets['D'] = 0
            granular_targets['D+'] = 0
        
        target_distributions['granular_targets'] = granular_targets
        
        print(f"Jury {jury_number} granular targets: P+={granular_targets['P+']}, P={granular_targets['P']}, D={granular_targets['D']}, D+={granular_targets['D+']}")
    
    # Calculate gender targets
    if 'Gender' in remaining_df.columns:
        male_count = remaining_df['Gender'].isin(['M', 'Male', 'male', 'MALE']).sum()
        female_count = remaining_df['Gender'].isin(['F', 'Female', 'female', 'FEMALE']).sum()
        
        # Try for balanced gender
        if jury_size % 2 == 0:
            ideal_male = jury_size // 2
            ideal_female = jury_size // 2
        else:
            # Give extra to gender with more availability
            if male_count >= female_count:
                ideal_male = (jury_size + 1) // 2
                ideal_female = jury_size // 2
            else:
                ideal_male = jury_size // 2
                ideal_female = (jury_size + 1) // 2
        
        # Adjust for availability
        if male_count < ideal_male:
            actual_male = male_count
            actual_female = jury_size - actual_male
        elif female_count < ideal_female:
            actual_female = female_count
            actual_male = jury_size - actual_female
        else:
            actual_male = ideal_male
            actual_female = ideal_female
        
        target_distributions['male_target'] = actual_male
        target_distributions['female_target'] = actual_female
        
        print(f"Jury {jury_number} gender targets: {actual_male}M, {actual_female}F")
    
    return target_distributions


def optimize_jury_assignment(data_dict, priority_weights=None, time_limit=300):
    """
    Main function to perform sequential jury assignment optimization.
    Optimizes juries one at a time to ensure complete jury filling.
    
    Parameters:
    data_dict (dict): Dictionary containing processed juror data from data_processing
    priority_weights (dict, optional): Dictionary mapping demographic variables to priority weights
    time_limit (int, optional): Time limit for solver in seconds per jury
    
    Returns:
    dict: Optimization results with sequential optimization information
    """
    print("Starting sequential jury optimization...")
    print("Approach: Optimize juries one at a time to guarantee complete filling")
    print("Priority: Jury A gets best balance, Jury B gets best possible balance from remaining jurors, etc.")
    
    num_juries = data_dict['num_juries']
    jury_size = data_dict['jury_size']
    df = data_dict['original_data']
    
    # Initialize tracking variables
    all_assignments = []
    remaining_juror_indices = df.index.tolist()
    jury_results = {}
    
    # Sequential optimization: one jury at a time
    for jury_num in range(1, num_juries + 1):
        print(f"\n=== OPTIMIZING JURY {jury_num} ===")
        print(f"Remaining jurors: {len(remaining_juror_indices)}")
        
        if len(remaining_juror_indices) < jury_size:
            print(f"WARNING: Only {len(remaining_juror_indices)} jurors remaining, cannot fill Jury {jury_num} (needs {jury_size})")
            break
        
        # Calculate target distributions for this jury based on remaining jurors
        target_distributions = calculate_target_distributions_for_jury(
            remaining_juror_indices, jury_size, data_dict, jury_num, num_juries
        )
        
        # Create optimization model for this jury
        model, model_info = create_single_jury_optimization_model(
            data_dict, jury_num, remaining_juror_indices, target_distributions, priority_weights
        )
        
        # Solve optimization for this jury
        solution_info = solve_single_jury_optimization(model, model_info, time_limit)
        
        # Extract assignments for this jury
        if solution_info['status'] == 'Optimal':
            jury_assignment = extract_single_jury_assignments(solution_info, data_dict)
            
            # Store results for this jury
            jury_results[jury_num] = {
                'assigned_indices': jury_assignment['assigned_indices'],
                'solution_status': jury_assignment['solution_status'],
                'objective_value': jury_assignment['objective_value'],
                'target_distributions': target_distributions
            }
            
            # Add assignments to master list
            for idx in jury_assignment['assigned_indices']:
                all_assignments.append({
                    'juror_index': idx,
                    'jury': jury_num
                })
            
            # Remove assigned jurors from remaining pool
            remaining_juror_indices = jury_assignment['unassigned_indices']
            
            print(f"Jury {jury_num} optimization successful: {len(jury_assignment['assigned_indices'])} jurors assigned")
        else:
            print(f"ERROR: Jury {jury_num} optimization failed with status: {solution_info['status']}")
            break
    
    # Handle any remaining unassigned jurors
    for idx in remaining_juror_indices:
        all_assignments.append({
            'juror_index': idx,
            'jury': 'Unassigned'
        })
    
    print(f"\n=== SEQUENTIAL OPTIMIZATION COMPLETE ===")
    print(f"Successfully filled {len(jury_results)} out of {num_juries} juries")
    print(f"Remaining unassigned jurors: {len(remaining_juror_indices)}")
    
    # Create combined results in the expected format
    assignments_df = pd.DataFrame(all_assignments)
    juror_data = df.reset_index()
    juror_data = juror_data.rename(columns={'index': 'juror_index'})
    jury_assignments = pd.merge(assignments_df, juror_data, on='juror_index')
    
    # Analyze jury composition
    jury_analysis = {}
    overall_balance_achieved = {
        'sequential_optimization': True,
        'complete_juries_filled': len(jury_results),
        'target_juries': num_juries,
        'jury_details': {}
    }
    
    for jury_num in range(1, num_juries + 1):
        jury_j = jury_assignments[jury_assignments['jury'] == jury_num]
        
        if len(jury_j) > 0:
            # Basic analysis
            analysis = {
                'size': len(jury_j),
                'jurors': jury_j.to_dict('records')
            }
            
            # Add leaning analysis
            if 'Final_Leaning' in jury_j.columns:
                leaning_counts = jury_j['Final_Leaning'].value_counts().to_dict()
                analysis['leaning'] = leaning_counts
                
                # Check if targets were met
                if jury_num in jury_results:
                    targets = jury_results[jury_num]['target_distributions']
                    p_actual = sum(count for leaning, count in leaning_counts.items() if leaning in ['P', 'P+'])
                    d_actual = sum(count for leaning, count in leaning_counts.items() if leaning in ['D', 'D+'])
                    
                    p_target = targets.get('p_overall_target', 0)
                    d_target = targets.get('d_overall_target', 0)
                    
                    overall_balance_achieved['jury_details'][f'jury_{jury_num}_pd_achieved'] = (p_actual == p_target and d_actual == d_target)
            
            # Add gender analysis
            if 'Gender' in jury_j.columns:
                analysis['gender'] = jury_j['Gender'].value_counts().to_dict()
            
            # Add other demographics
            if 'Race' in jury_j.columns:
                analysis['race'] = jury_j['Race'].value_counts().to_dict()
            if 'AgeGroup' in jury_j.columns:
                analysis['age_group'] = jury_j['AgeGroup'].value_counts().to_dict()
            if 'Education' in jury_j.columns:
                analysis['education'] = jury_j['Education'].value_counts().to_dict()
            if 'Marital' in jury_j.columns:
                analysis['marital'] = jury_j['Marital'].value_counts().to_dict()
            
            jury_analysis[jury_num] = analysis
    
    # Add unassigned analysis
    unassigned_jurors = jury_assignments[jury_assignments['jury'] == 'Unassigned']
    if len(unassigned_jurors) > 0:
        unassigned_analysis = {
            'size': len(unassigned_jurors),
            'jurors': unassigned_jurors.to_dict('records')
        }
        
        if 'Final_Leaning' in unassigned_jurors.columns:
            unassigned_analysis['leaning'] = unassigned_jurors['Final_Leaning'].value_counts().to_dict()
        if 'Gender' in unassigned_jurors.columns:
            unassigned_analysis['gender'] = unassigned_jurors['Gender'].value_counts().to_dict()
        if 'Race' in unassigned_jurors.columns:
            unassigned_analysis['race'] = unassigned_jurors['Race'].value_counts().to_dict()
        if 'AgeGroup' in unassigned_jurors.columns:
            unassigned_analysis['age_group'] = unassigned_jurors['AgeGroup'].value_counts().to_dict()
        if 'Education' in unassigned_jurors.columns:
            unassigned_analysis['education'] = unassigned_jurors['Education'].value_counts().to_dict()
        if 'Marital' in unassigned_jurors.columns:
            unassigned_analysis['marital'] = unassigned_jurors['Marital'].value_counts().to_dict()
            
        jury_analysis['Unassigned'] = unassigned_analysis
    
    # Calculate deviations (simplified for sequential approach)
    total_deviations = {}
    for jury_num in jury_results:
        if 'objective_value' in jury_results[jury_num] and jury_results[jury_num]['objective_value']:
            total_deviations[f'Jury_{jury_num}'] = jury_results[jury_num]['objective_value']
    
    # Format results for output
    formatted_results = format_results_for_output({
        'assignments': jury_assignments,
        'jury_analysis': jury_analysis,
        'deviations': total_deviations,
        'solution_status': 'Sequential Optimization Complete',
        'objective_value': sum(total_deviations.values()) if total_deviations else 0,
        'unassigned_count': len(unassigned_jurors),
        'balance_achieved': overall_balance_achieved,
        'sequential_results': jury_results
    }, data_dict)
    
    return formatted_results


def format_results_for_output(assignment_results, data_dict):
    """
    Format the assignment results for output with sequential optimization reporting.
    """
    jury_assignments = assignment_results['assignments']
    jury_analysis = assignment_results['jury_analysis']
    
    # Create a summary DataFrame for each jury
    jury_summaries = []
    
    for jury_num, analysis in jury_analysis.items():
        # Handle both numeric and alphabetic jury IDs, skip unassigned
        if jury_num == 'Unassigned':
            continue
            
        try:
            jury_label = chr(64 + int(jury_num))
        except (ValueError, TypeError):
            jury_label = str(jury_num)
        
        # Count P and D leanings (overall and granular)
        leaning_key = 'leaning' if 'leaning' in analysis else None
        
        summary = {
            'Jury': jury_label,
            'Size': analysis['size'],
        }
        
        # Add P/D leaning counts if available
        if leaning_key:
            # Overall P/D counts
            p_overall_count = sum(count for leaning, count in analysis[leaning_key].items() 
                        if leaning in ['P', 'P+'])
            d_overall_count = sum(count for leaning, count in analysis[leaning_key].items() 
                        if leaning in ['D', 'D+'])
            
            # Legacy fields
            summary['P_Leaning'] = p_overall_count
            summary['D_Leaning'] = d_overall_count
            summary['P_D_Ratio'] = f"{p_overall_count}:{d_overall_count}"
            
            # New overall fields
            summary['P_Overall'] = p_overall_count
            summary['D_Overall'] = d_overall_count
            
            # Granular counts
            summary['P+_Count'] = analysis[leaning_key].get('P+', 0)
            summary['P_Count'] = analysis[leaning_key].get('P', 0)
            summary['D_Count'] = analysis[leaning_key].get('D', 0)
            summary['D+_Count'] = analysis[leaning_key].get('D+', 0)
        else:
            # Default values
            summary['P_Leaning'] = 0
            summary['D_Leaning'] = 0
            summary['P_D_Ratio'] = "0:0"
            summary['P_Overall'] = 0
            summary['D_Overall'] = 0
            summary['P+_Count'] = 0
            summary['P_Count'] = 0
            summary['D_Count'] = 0
            summary['D+_Count'] = 0

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
            for edu, count in analysis['education'].items():
                summary[f'Education_{edu}'] = count
        
        # Add marital distribution
        if 'marital' in analysis and analysis['marital']:
            for marital, count in analysis['marital'].items():
                summary[f'Marital_{marital}'] = count
        
        jury_summaries.append(summary)
    
    summary_df = pd.DataFrame(jury_summaries)
    
    # Format detailed assignments
    detailed_assignments = jury_assignments.sort_values(['jury', 'Name'])
    
    # Select relevant columns
    output_columns = ['jury', 'Name', '#', 'Final_Leaning', 'Gender', 'Race', 'Age', 'AgeGroup', 'Education', 'Marital']
    output_columns = [col for col in output_columns if col in detailed_assignments.columns]
    
    detailed_output = detailed_assignments[output_columns].copy()
    # Add alphabetic jury ID column
    detailed_output['jury_letter'] = detailed_output['jury'].apply(
        lambda x: chr(64 + int(x)) if isinstance(x, (int, float)) and x != 'Unassigned' else x
    )
    
    # Replace numeric with alphabetic (except Unassigned)
    detailed_output['jury'] = detailed_output['jury_letter']
    detailed_output = detailed_output.drop('jury_letter', axis=1)

    # Calculate overall balance metrics with sequential optimization info
    balance_metrics = {}
    
    # Overall leaning balance
    leaning_col = 'Final_Leaning'
    if leaning_col in detailed_assignments.columns:
        assigned_assignments = detailed_assignments[detailed_assignments['jury'] != 'Unassigned']
        
        overall_p = assigned_assignments[leaning_col].isin(['P', 'P+']).sum()
        overall_d = assigned_assignments[leaning_col].isin(['D', 'D+']).sum()
        
        # Granular counts
        granular_counts = {
            'P+': (assigned_assignments[leaning_col] == 'P+').sum(),
            'P': (assigned_assignments[leaning_col] == 'P').sum(),
            'D': (assigned_assignments[leaning_col] == 'D').sum(),
            'D+': (assigned_assignments[leaning_col] == 'D+').sum()
        }
        
        leaning_balance = {
            'overall_p': overall_p,
            'overall_d': overall_d,
            'p_percentage': round(overall_p / len(assigned_assignments) * 100, 2) if len(assigned_assignments) > 0 else 0,
            'granular_counts': granular_counts,
            'sequential_optimization': True,
            'complete_juries_achieved': assignment_results.get('balance_achieved', {}).get('complete_juries_filled', 0)
        }
        
        balance_metrics['leaning'] = leaning_balance
    
    # Gender balance
    if 'Gender' in detailed_assignments.columns:
        assigned_assignments = detailed_assignments[detailed_assignments['jury'] != 'Unassigned']
        
        overall_male = assigned_assignments['Gender'].isin(['M', 'Male', 'male', 'MALE']).sum()
        overall_female = assigned_assignments['Gender'].isin(['F', 'Female', 'female', 'FEMALE']).sum()
        
        gender_balance = {
            'overall_male': overall_male,
            'overall_female': overall_female,
            'male_percentage': round(overall_male / len(assigned_assignments) * 100, 2) if len(assigned_assignments) > 0 else 0,
            'sequential_optimization': True
        }
        
        balance_metrics['gender'] = gender_balance
    
    # Return formatted results with sequential optimization information
    return {
        'summary': summary_df,
        'detailed_assignments': detailed_output,
        'balance_metrics': balance_metrics,
        'jury_analysis': jury_analysis,
        'solution_quality': {
            'status': assignment_results['solution_status'],
            'objective_value': assignment_results['objective_value'],
            'deviations': assignment_results['deviations'],
            'sequential_balance': assignment_results.get('balance_achieved', {}),
            'sequential_results': assignment_results.get('sequential_results', {}),
            'optimization_method': 'Sequential Jury Optimization'
        }
    }


if __name__ == "__main__":
    # Example usage (for testing)
    from data_processing import process_juror_data
    
    file_path = r"C:\Users\NicholasWilson\OneDrive - Trial Behavior Consulting\AutoJurySplit_Data.xlsx"
    num_juries = 2
    jury_size = 12
    
    try:
        # Process the data
        data_dict = process_juror_data(file_path, num_juries, jury_size)
        
        # Define priority weights (only affects tertiary level)
        priority_weights = {
            'Race': 3.0,           # TIER 3: Weighted optimization
            'AgeGroup': 2.0,       # TIER 3: Weighted optimization
            'Education': 1.0,      # TIER 3: Weighted optimization
            'Marital': 0.5         # TIER 3: Weighted optimization
        }
        
        # Run sequential optimization
        results = optimize_jury_assignment(data_dict, priority_weights)
        
        # Print summary
        print("\nSequential Jury Assignment Summary:")
        print(results['summary'])
        
        # Print solution quality
        print("\nSolution Quality:")
        solution_quality = results['solution_quality']
        print(f"Status: {solution_quality['status']}")
        print(f"Method: {solution_quality['optimization_method']}")
        print(f"Objective Value: {solution_quality['objective_value']}")
        
        # Print sequential balance results
        if 'sequential_balance' in solution_quality:
            balance = solution_quality['sequential_balance']
            print(f"Complete juries filled: {balance.get('complete_juries_filled', 0)} out of {balance.get('target_juries', 0)}")
        
        # Print balance metrics
        if 'leaning' in results['balance_metrics']:
            leaning = results['balance_metrics']['leaning']
            print(f"\nOverall Leaning Balance: {leaning['overall_p']} P : {leaning['overall_d']} D")
            if 'granular_counts' in leaning:
                print(f"Granular Leaning Counts: P+={leaning['granular_counts']['P+']}, P={leaning['granular_counts']['P']}, D={leaning['granular_counts']['D']}, D+={leaning['granular_counts']['D+']}")
            
        if 'gender' in results['balance_metrics']:
            gender = results['balance_metrics']['gender']
            print(f"Overall Gender Balance: {gender['overall_male']} M : {gender['overall_female']} F")
        
    except Exception as e:
        import traceback
        print(f"Error in sequential optimization process: {str(e)}")
        print(traceback.format_exc())