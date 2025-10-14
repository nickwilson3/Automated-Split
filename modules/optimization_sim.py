import numpy as np
import pandas as pd
from pulp import *
from itertools import combinations


def create_simultaneous_optimization_model(data_dict, tier_weights=None):
    """
    Create optimization model for all juries simultaneously.
    Minimizes differences between juries across all demographics.
    
    Parameters:
    data_dict (dict): Dictionary containing processed juror data and parameters
    tier_weights (dict, optional): Weights for different tiers of constraints
    
    Returns:
    tuple: (model, model_info)
    """
    # Extract data from the dictionary
    df = data_dict['original_data']
    num_juries = data_dict['num_juries']
    jury_size = data_dict['jury_size']
    category_counts = data_dict['category_counts']
    demographic_vars = data_dict['demographic_vars']
    
    # Set default tier weights if not provided
    if tier_weights is None:
        tier_weights = {
            'tier1_pd_balance': 1000.0,
            'tier2_granular_leaning': 100.0,
            'tier2_gender': 90.0,
            'tier3_race': 5.0,
            'tier3_age': 3.0,
            'tier3_education': 2.0,
            'tier3_marital': 1.0
        }
    
    print(f"Creating simultaneous optimization model for {num_juries} juries")
    print(f"Tier weights: {tier_weights}")
    
    # Create the optimization model
    model = LpProblem("Simultaneous_Jury_Optimization", LpMinimize)
    
    # Define decision variables
    # x[i][j] = 1 if juror i is assigned to jury j
    juror_indices = df.index.tolist()
    jury_ids = list(range(1, num_juries + 1))
    
    x = {}
    for i in juror_indices:
        for j in jury_ids:
            x[(i, j)] = LpVariable(f"Assign_Juror_{i}_to_Jury_{j}", cat=LpBinary)
    
    # Define deviation variables for measuring differences between juries
    deviation_vars = {}
    objective_terms = []
    
    # ========== HARD CONSTRAINTS ==========
    
    print("Adding hard constraints...")
    
    # Constraint 1: Each jury must have exactly jury_size jurors
    for j in jury_ids:
        model += lpSum(x[(i, j)] for i in juror_indices) == jury_size, f"Jury_{j}_Size"
        print(f"Hard constraint: Jury {j} must have exactly {jury_size} jurors")
    
    # Constraint 2: Each juror assigned to at most one jury
    for i in juror_indices:
        model += lpSum(x[(i, j)] for j in jury_ids) <= 1, f"Juror_{i}_Assignment"
    
    print(f"Hard constraint: Each juror assigned to at most one jury")
    
    # ========== TIER 1: P/D OVERALL BALANCE SIMILARITY ==========
    
    print("\nAdding TIER 1 constraints (P/D overall balance similarity)...")
    
    if 'Final_Leaning' in demographic_vars and 'Final_Leaning' in df.columns:
        # Get indices for P-leaning and D-leaning jurors
        p_overall_indices = df[df['Final_Leaning'].isin(['P', 'P+'])].index.tolist()
        d_overall_indices = df[df['Final_Leaning'].isin(['D', 'D+'])].index.tolist()
        
        # For each pair of juries, minimize difference in P and D counts
        for j1, j2 in combinations(jury_ids, 2):
            # P-leaning difference
            if p_overall_indices:
                p_diff_plus = LpVariable(f"P_Overall_Diff_Plus_J{j1}_J{j2}", lowBound=0)
                p_diff_minus = LpVariable(f"P_Overall_Diff_Minus_J{j1}_J{j2}", lowBound=0)
                
                p_count_j1 = lpSum(x[(i, j1)] for i in p_overall_indices)
                p_count_j2 = lpSum(x[(i, j2)] for i in p_overall_indices)
                
                model += p_count_j1 - p_count_j2 - p_diff_plus + p_diff_minus == 0, f"P_Overall_Balance_J{j1}_J{j2}"
                
                # Add to objective with high weight
                objective_terms.append(tier_weights['tier1_pd_balance'] * (p_diff_plus + p_diff_minus))
                
                deviation_vars[f'P_Overall_J{j1}_J{j2}'] = {'plus': p_diff_plus, 'minus': p_diff_minus}
            
            # D-leaning difference
            if d_overall_indices:
                d_diff_plus = LpVariable(f"D_Overall_Diff_Plus_J{j1}_J{j2}", lowBound=0)
                d_diff_minus = LpVariable(f"D_Overall_Diff_Minus_J{j1}_J{j2}", lowBound=0)
                
                d_count_j1 = lpSum(x[(i, j1)] for i in d_overall_indices)
                d_count_j2 = lpSum(x[(i, j2)] for i in d_overall_indices)
                
                model += d_count_j1 - d_count_j2 - d_diff_plus + d_diff_minus == 0, f"D_Overall_Balance_J{j1}_J{j2}"
                
                # Add to objective with high weight
                objective_terms.append(tier_weights['tier1_pd_balance'] * (d_diff_plus + d_diff_minus))
                
                deviation_vars[f'D_Overall_J{j1}_J{j2}'] = {'plus': d_diff_plus, 'minus': d_diff_minus}
        
        print(f"TIER 1: Added P/D overall balance similarity constraints for all jury pairs")
    
    # ========== TIER 2: GRANULAR LEANING SIMILARITY ==========
    
    print("\nAdding TIER 2 constraints (granular P+/P/D/D+ similarity)...")
    
    if 'Final_Leaning' in demographic_vars and 'Final_Leaning' in df.columns:
        granular_categories = ['P+', 'P', 'D', 'D+']
        
        for category in granular_categories:
            cat_indices = df[df['Final_Leaning'] == category].index.tolist()
            
            if not cat_indices:
                continue
            
            # For each pair of juries, minimize difference in this category
            for j1, j2 in combinations(jury_ids, 2):
                cat_diff_plus = LpVariable(f"{category}_Diff_Plus_J{j1}_J{j2}", lowBound=0)
                cat_diff_minus = LpVariable(f"{category}_Diff_Minus_J{j1}_J{j2}", lowBound=0)
                
                cat_count_j1 = lpSum(x[(i, j1)] for i in cat_indices)
                cat_count_j2 = lpSum(x[(i, j2)] for i in cat_indices)
                
                model += cat_count_j1 - cat_count_j2 - cat_diff_plus + cat_diff_minus == 0, f"{category}_Balance_J{j1}_J{j2}"
                
                # Add to objective with medium-high weight
                objective_terms.append(tier_weights['tier2_granular_leaning'] * (cat_diff_plus + cat_diff_minus))
                
                deviation_vars[f'{category}_J{j1}_J{j2}'] = {'plus': cat_diff_plus, 'minus': cat_diff_minus}
        
        print(f"TIER 2: Added granular leaning similarity constraints for all jury pairs")
    
    # ========== TIER 2: GENDER SIMILARITY ==========
    
    print("\nAdding TIER 2 constraints (gender similarity)...")
    
    if 'Gender' in demographic_vars and 'Gender' in df.columns:
        male_indices = df[df['Gender'].isin(['M', 'Male', 'male', 'MALE'])].index.tolist()
        female_indices = df[df['Gender'].isin(['F', 'Female', 'female', 'FEMALE'])].index.tolist()
        
        # For each pair of juries, minimize gender differences
        for j1, j2 in combinations(jury_ids, 2):
            # Male difference
            if male_indices:
                male_diff_plus = LpVariable(f"Male_Diff_Plus_J{j1}_J{j2}", lowBound=0)
                male_diff_minus = LpVariable(f"Male_Diff_Minus_J{j1}_J{j2}", lowBound=0)
                
                male_count_j1 = lpSum(x[(i, j1)] for i in male_indices)
                male_count_j2 = lpSum(x[(i, j2)] for i in male_indices)
                
                model += male_count_j1 - male_count_j2 - male_diff_plus + male_diff_minus == 0, f"Male_Balance_J{j1}_J{j2}"
                
                objective_terms.append(tier_weights['tier2_gender'] * (male_diff_plus + male_diff_minus))
                
                deviation_vars[f'Male_J{j1}_J{j2}'] = {'plus': male_diff_plus, 'minus': male_diff_minus}
            
            # Female difference
            if female_indices:
                female_diff_plus = LpVariable(f"Female_Diff_Plus_J{j1}_J{j2}", lowBound=0)
                female_diff_minus = LpVariable(f"Female_Diff_Minus_J{j1}_J{j2}", lowBound=0)
                
                female_count_j1 = lpSum(x[(i, j1)] for i in female_indices)
                female_count_j2 = lpSum(x[(i, j2)] for i in female_indices)
                
                model += female_count_j1 - female_count_j2 - female_diff_plus + female_diff_minus == 0, f"Female_Balance_J{j1}_J{j2}"
                
                objective_terms.append(tier_weights['tier2_gender'] * (female_diff_plus + female_diff_minus))
                
                deviation_vars[f'Female_J{j1}_J{j2}'] = {'plus': female_diff_plus, 'minus': female_diff_minus}
        
        print(f"TIER 2: Added gender similarity constraints for all jury pairs")
    
    # ========== TIER 3: OTHER DEMOGRAPHIC SIMILARITIES ==========
    
    print("\nAdding TIER 3 constraints (other demographics similarity)...")
    
    # Process Race
    if 'Race' in demographic_vars and 'Race' in df.columns and 'Race' in category_counts:
        race_categories = df['Race'].dropna().unique()
        
        for race in race_categories:
            race_indices = df[df['Race'] == race].index.tolist()
            
            if not race_indices:
                continue
            
            for j1, j2 in combinations(jury_ids, 2):
                race_diff_plus = LpVariable(f"Race_{race}_Diff_Plus_J{j1}_J{j2}", lowBound=0)
                race_diff_minus = LpVariable(f"Race_{race}_Diff_Minus_J{j1}_J{j2}", lowBound=0)
                
                race_count_j1 = lpSum(x[(i, j1)] for i in race_indices)
                race_count_j2 = lpSum(x[(i, j2)] for i in race_indices)
                
                model += race_count_j1 - race_count_j2 - race_diff_plus + race_diff_minus == 0, f"Race_{race}_Balance_J{j1}_J{j2}"
                
                objective_terms.append(tier_weights['tier3_race'] * (race_diff_plus + race_diff_minus))
                
                if f'Race_{race}' not in deviation_vars:
                    deviation_vars[f'Race_{race}'] = {}
                deviation_vars[f'Race_{race}'][f'J{j1}_J{j2}'] = {'plus': race_diff_plus, 'minus': race_diff_minus}
        
        print(f"TIER 3: Added race similarity constraints")
    
    # Process Age Groups
    if 'AgeGroup' in df.columns:
        age_categories = df['AgeGroup'].dropna().unique()
        
        for age in age_categories:
            age_indices = df[df['AgeGroup'] == age].index.tolist()
            
            if not age_indices:
                continue
            
            for j1, j2 in combinations(jury_ids, 2):
                age_diff_plus = LpVariable(f"Age_{age}_Diff_Plus_J{j1}_J{j2}", lowBound=0)
                age_diff_minus = LpVariable(f"Age_{age}_Diff_Minus_J{j1}_J{j2}", lowBound=0)
                
                age_count_j1 = lpSum(x[(i, j1)] for i in age_indices)
                age_count_j2 = lpSum(x[(i, j2)] for i in age_indices)
                
                model += age_count_j1 - age_count_j2 - age_diff_plus + age_diff_minus == 0, f"Age_{age}_Balance_J{j1}_J{j2}"
                
                objective_terms.append(tier_weights['tier3_age'] * (age_diff_plus + age_diff_minus))
                
                if f'Age_{age}' not in deviation_vars:
                    deviation_vars[f'Age_{age}'] = {}
                deviation_vars[f'Age_{age}'][f'J{j1}_J{j2}'] = {'plus': age_diff_plus, 'minus': age_diff_minus}
        
        print(f"TIER 3: Added age group similarity constraints")
    
    # Process Education
    if 'Education' in demographic_vars and 'Education' in df.columns and 'Education' in category_counts:
        education_categories = df['Education'].dropna().unique()
        
        for edu in education_categories:
            edu_indices = df[df['Education'] == edu].index.tolist()
            
            if not edu_indices:
                continue
            
            for j1, j2 in combinations(jury_ids, 2):
                edu_diff_plus = LpVariable(f"Edu_{edu}_Diff_Plus_J{j1}_J{j2}", lowBound=0)
                edu_diff_minus = LpVariable(f"Edu_{edu}_Diff_Minus_J{j1}_J{j2}", lowBound=0)
                
                edu_count_j1 = lpSum(x[(i, j1)] for i in edu_indices)
                edu_count_j2 = lpSum(x[(i, j2)] for i in edu_indices)
                
                model += edu_count_j1 - edu_count_j2 - edu_diff_plus + edu_diff_minus == 0, f"Edu_{edu}_Balance_J{j1}_J{j2}"
                
                objective_terms.append(tier_weights['tier3_education'] * (edu_diff_plus + edu_diff_minus))
                
                if f'Education_{edu}' not in deviation_vars:
                    deviation_vars[f'Education_{edu}'] = {}
                deviation_vars[f'Education_{edu}'][f'J{j1}_J{j2}'] = {'plus': edu_diff_plus, 'minus': edu_diff_minus}
        
        print(f"TIER 3: Added education similarity constraints")
    
    # Process Marital Status
    if 'Marital' in demographic_vars and 'Marital' in df.columns and 'Marital' in category_counts:
        marital_categories = df['Marital'].dropna().unique()
        
        for marital in marital_categories:
            marital_indices = df[df['Marital'] == marital].index.tolist()
            
            if not marital_indices:
                continue
            
            for j1, j2 in combinations(jury_ids, 2):
                marital_diff_plus = LpVariable(f"Marital_{marital}_Diff_Plus_J{j1}_J{j2}", lowBound=0)
                marital_diff_minus = LpVariable(f"Marital_{marital}_Diff_Minus_J{j1}_J{j2}", lowBound=0)
                
                marital_count_j1 = lpSum(x[(i, j1)] for i in marital_indices)
                marital_count_j2 = lpSum(x[(i, j2)] for i in marital_indices)
                
                model += marital_count_j1 - marital_count_j2 - marital_diff_plus + marital_diff_minus == 0, f"Marital_{marital}_Balance_J{j1}_J{j2}"
                
                objective_terms.append(tier_weights['tier3_marital'] * (marital_diff_plus + marital_diff_minus))
                
                if f'Marital_{marital}' not in deviation_vars:
                    deviation_vars[f'Marital_{marital}'] = {}
                deviation_vars[f'Marital_{marital}'][f'J{j1}_J{j2}'] = {'plus': marital_diff_plus, 'minus': marital_diff_minus}
        
        print(f"TIER 3: Added marital status similarity constraints")
    
    # ========== OBJECTIVE FUNCTION ==========
    
    if objective_terms:
        model += lpSum(objective_terms)
        print(f"\nObjective function includes {len(objective_terms)} weighted deviation terms")
    else:
        # If no objective terms, minimize a dummy variable
        dummy_var = LpVariable("Dummy_Objective", lowBound=0)
        model += dummy_var
        model += dummy_var == 0
        print(f"\nUsing dummy objective (no soft constraints)")
    
    # Return the model and important variables
    model_info = {
        'model': model,
        'x': x,
        'deviation_vars': deviation_vars,
        'juror_indices': juror_indices,
        'jury_ids': jury_ids,
        'num_juries': num_juries,
        'jury_size': jury_size,
        'tier_weights': tier_weights
    }
    
    return model, model_info


def solve_simultaneous_optimization(model, model_info, time_limit=600, gap_tolerance=0.01):
    """
    Solve the simultaneous optimization model.
    
    Parameters:
    model (LpProblem): The optimization model
    model_info (dict): Dictionary containing model variables and information
    time_limit (int, optional): Time limit for solver in seconds
    gap_tolerance (float, optional): Optimality gap tolerance (e.g., 0.01 = 1%)
    
    Returns:
    dict: Solution information
    """
    print(f"\nSolving simultaneous optimization model...")
    print(f"Time limit: {time_limit} seconds")
    print(f"Gap tolerance: {gap_tolerance * 100}%")
    
    # Set up solver with options
    solver = PULP_CBC_CMD(
        timeLimit=time_limit,
        gapRel=gap_tolerance,
        msg=1  # Show solver output
    )
    
    # Solve the model
    model.solve(solver)
    
    # Check solution status
    status = LpStatus[model.status]
    print(f"\nSolution status: {status}")
    
    if status == 'Optimal':
        print(f"Objective value: {value(model.objective):.2f}")
        print("Optimal solution found!")
    elif status == 'Infeasible':
        print("ERROR: Optimization is infeasible")
        print("This indicates hard constraints cannot be satisfied")
    elif status in ['Not Solved', 'Undefined']:
        print(f"WARNING: Solver did not find optimal solution: {status}")
    
    return {
        'status': status,
        'objective_value': value(model.objective) if status in ['Optimal', 'Feasible'] else None,
        'model': model,
        'model_info': model_info
    }


def extract_jury_assignments(solution_info, data_dict):
    """
    Extract jury assignments from the simultaneous optimization solution.
    
    Parameters:
    solution_info (dict): Solution information from solve_simultaneous_optimization
    data_dict (dict): Dictionary containing processed juror data
    
    Returns:
    dict: Assignment results
    """
    model_info = solution_info['model_info']
    df = data_dict['original_data']
    
    x = model_info['x']
    juror_indices = model_info['juror_indices']
    jury_ids = model_info['jury_ids']
    
    # Extract assignments
    assignments = []
    
    for i in juror_indices:
        assigned = False
        for j in jury_ids:
            if value(x[(i, j)]) > 0.5:  # Binary variable is 1 (assigned)
                assignments.append({
                    'juror_index': i,
                    'jury': j
                })
                assigned = True
                break
        
        if not assigned:
            # Juror is unassigned
            assignments.append({
                'juror_index': i,
                'jury': 'Unassigned'
            })
    
    # Create DataFrame with assignments
    assignments_df = pd.DataFrame(assignments)
    
    # Merge with original juror data
    juror_data = df.reset_index()
    juror_data = juror_data.rename(columns={'index': 'juror_index'})
    jury_assignments = pd.merge(assignments_df, juror_data, on='juror_index')
    
    # Count assignments per jury
    for j in jury_ids:
        count = (jury_assignments['jury'] == j).sum()
        print(f"Jury {j}: {count} jurors assigned")
    
    unassigned_count = (jury_assignments['jury'] == 'Unassigned').sum()
    if unassigned_count > 0:
        print(f"Unassigned: {unassigned_count} jurors")
    
    return {
        'assignments': jury_assignments,
        'solution_status': solution_info['status'],
        'objective_value': solution_info['objective_value']
    }


def analyze_jury_composition(assignments_df, jury_ids):
    """
    Analyze the composition of each jury.
    
    Parameters:
    assignments_df (DataFrame): Jury assignments with juror data
    jury_ids (list): List of jury IDs
    
    Returns:
    dict: Jury analysis for each jury
    """
    jury_analysis = {}
    
    for j in jury_ids:
        jury_j = assignments_df[assignments_df['jury'] == j]
        
        if len(jury_j) == 0:
            continue
        
        # Basic analysis
        analysis = {
            'size': len(jury_j),
            'jurors': jury_j.to_dict('records')
        }
        
        # Add leaning analysis
        if 'Final_Leaning' in jury_j.columns:
            leaning_counts = jury_j['Final_Leaning'].value_counts().to_dict()
            analysis['leaning'] = leaning_counts
        
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
        
        jury_analysis[j] = analysis
    
    # Add unassigned analysis
    unassigned = assignments_df[assignments_df['jury'] == 'Unassigned']
    if len(unassigned) > 0:
        unassigned_analysis = {
            'size': len(unassigned),
            'jurors': unassigned.to_dict('records')
        }
        
        if 'Final_Leaning' in unassigned.columns:
            unassigned_analysis['leaning'] = unassigned['Final_Leaning'].value_counts().to_dict()
        if 'Gender' in unassigned.columns:
            unassigned_analysis['gender'] = unassigned['Gender'].value_counts().to_dict()
        if 'Race' in unassigned.columns:
            unassigned_analysis['race'] = unassigned['Race'].value_counts().to_dict()
        if 'AgeGroup' in unassigned.columns:
            unassigned_analysis['age_group'] = unassigned['AgeGroup'].value_counts().to_dict()
        if 'Education' in unassigned.columns:
            unassigned_analysis['education'] = unassigned['Education'].value_counts().to_dict()
        if 'Marital' in unassigned.columns:
            unassigned_analysis['marital'] = unassigned['Marital'].value_counts().to_dict()
        
        jury_analysis['Unassigned'] = unassigned_analysis
    
    return jury_analysis


def calculate_similarity_metrics(jury_analysis, jury_ids):
    """
    Calculate similarity metrics between juries.
    
    Parameters:
    jury_analysis (dict): Analysis for each jury
    jury_ids (list): List of jury IDs
    
    Returns:
    dict: Similarity metrics
    """
    similarity_metrics = {
        'pairwise_differences': {},
        'max_difference': {},
        'similarity_score': {}
    }
    
    # Calculate pairwise differences for P/D leaning
    if all('leaning' in jury_analysis.get(j, {}) for j in jury_ids):
        for j1, j2 in combinations(jury_ids, 2):
            leaning_j1 = jury_analysis[j1]['leaning']
            leaning_j2 = jury_analysis[j2]['leaning']
            
            p_j1 = sum(count for leaning, count in leaning_j1.items() if leaning in ['P', 'P+'])
            p_j2 = sum(count for leaning, count in leaning_j2.items() if leaning in ['P', 'P+'])
            
            d_j1 = sum(count for leaning, count in leaning_j1.items() if leaning in ['D', 'D+'])
            d_j2 = sum(count for leaning, count in leaning_j2.items() if leaning in ['D', 'D+'])
            
            similarity_metrics['pairwise_differences'][f'P_J{j1}_J{j2}'] = abs(p_j1 - p_j2)
            similarity_metrics['pairwise_differences'][f'D_J{j1}_J{j2}'] = abs(d_j1 - d_j2)
    
    # Calculate max differences across all juries for each demographic
    for demo in ['leaning', 'gender', 'race', 'age_group', 'education', 'marital']:
        if all(demo in jury_analysis.get(j, {}) for j in jury_ids):
            # Get all categories across all juries
            all_categories = set()
            for j in jury_ids:
                all_categories.update(jury_analysis[j][demo].keys())
            
            # For each category, find max difference
            for category in all_categories:
                counts = [jury_analysis[j][demo].get(category, 0) for j in jury_ids]
                max_diff = max(counts) - min(counts)
                similarity_metrics['max_difference'][f'{demo}_{category}'] = max_diff
    
    return similarity_metrics


def optimize_jury_assignment(data_dict, tier_weights=None, time_limit=600, gap_tolerance=0.01):
    """
    Main function to perform simultaneous jury assignment optimization.
    Optimizes all juries together to maximize similarity.
    
    Parameters:
    data_dict (dict): Dictionary containing processed juror data from data_processing
    tier_weights (dict, optional): Weights for different tiers of constraints
    time_limit (int, optional): Time limit for solver in seconds
    gap_tolerance (float, optional): Optimality gap tolerance
    
    Returns:
    dict: Optimization results with simultaneous optimization information
    """
    print("=" * 60)
    print("STARTING SIMULTANEOUS JURY OPTIMIZATION")
    print("=" * 60)
    print("Approach: Optimize all juries together to maximize similarity")
    print("Goal: Minimize differences in demographics between jury pairs")
    print("=" * 60)
    
    num_juries = data_dict['num_juries']
    jury_size = data_dict['jury_size']
    
    # Create optimization model
    model, model_info = create_simultaneous_optimization_model(data_dict, tier_weights)
    
    # Solve optimization
    solution_info = solve_simultaneous_optimization(model, model_info, time_limit, gap_tolerance)
    
    # Extract assignments
    if solution_info['status'] in ['Optimal', 'Feasible']:
        assignment_results = extract_jury_assignments(solution_info, data_dict)
        
        # Analyze jury composition
        jury_analysis = analyze_jury_composition(
            assignment_results['assignments'],
            model_info['jury_ids']
        )
        
        # Calculate similarity metrics
        similarity_metrics = calculate_similarity_metrics(jury_analysis, model_info['jury_ids'])
        
        # Calculate deviation summary from the model
        deviation_summary = {}
        if 'deviation_vars' in model_info:
            for var_name, dev_var in model_info['deviation_vars'].items():
                if isinstance(dev_var, dict) and 'plus' in dev_var and 'minus' in dev_var:
                    plus_val = value(dev_var['plus']) if dev_var['plus'] else 0
                    minus_val = value(dev_var['minus']) if dev_var['minus'] else 0
                    deviation_summary[var_name] = plus_val + minus_val
        
        print("\n" + "=" * 60)
        print("SIMULTANEOUS OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Status: {solution_info['status']}")
        print(f"Objective value: {solution_info['objective_value']:.2f}")
        print(f"All {num_juries} juries optimized simultaneously")
        print("=" * 60)
        
        # Format results for output
        formatted_results = format_results_for_output({
            'assignments': assignment_results['assignments'],
            'jury_analysis': jury_analysis,
            'deviations': deviation_summary,
            'similarity_metrics': similarity_metrics,
            'solution_status': solution_info['status'],
            'objective_value': solution_info['objective_value'],
            'tier_weights': model_info['tier_weights']
        }, data_dict)
        
        return formatted_results
    else:
        print("\n" + "=" * 60)
        print("OPTIMIZATION FAILED")
        print("=" * 60)
        print(f"Status: {solution_info['status']}")
        print("No feasible solution found")
        print("=" * 60)
        
        raise Exception(f"Optimization failed with status: {solution_info['status']}")


def format_results_for_output(assignment_results, data_dict):
    """
    Format the assignment results for output with simultaneous optimization reporting.
    
    Parameters:
    assignment_results (dict): Raw assignment results
    data_dict (dict): Processed juror data
    
    Returns:
    dict: Formatted results
    """
    jury_assignments = assignment_results['assignments']
    jury_analysis = assignment_results['jury_analysis']
    
    # Create a summary DataFrame for each jury
    jury_summaries = []
    
    for jury_num, analysis in jury_analysis.items():
        # Skip unassigned
        if jury_num == 'Unassigned':
            continue
        
        # Convert jury number to letter
        try:
            jury_label = chr(64 + int(jury_num))
        except (ValueError, TypeError):
            jury_label = str(jury_num)
        
        summary = {
            'Jury': jury_label,
            'Size': analysis['size'],
        }
        
        # Add P/D leaning counts if available
        if 'leaning' in analysis:
            # Overall P/D counts
            p_overall_count = sum(count for leaning, count in analysis['leaning'].items() 
                                if leaning in ['P', 'P+'])
            d_overall_count = sum(count for leaning, count in analysis['leaning'].items() 
                                if leaning in ['D', 'D+'])
            
            summary['P_Overall'] = p_overall_count
            summary['D_Overall'] = d_overall_count
            summary['P_D_Ratio'] = f"{p_overall_count}:{d_overall_count}"
            
            # Granular counts
            summary['P+_Count'] = analysis['leaning'].get('P+', 0)
            summary['P_Count'] = analysis['leaning'].get('P', 0)
            summary['D_Count'] = analysis['leaning'].get('D', 0)
            summary['D+_Count'] = analysis['leaning'].get('D+', 0)
        
        # Add gender distribution if available
        if 'gender' in analysis:
            for gender, count in analysis['gender'].items():
                summary[f'Gender_{gender}'] = count
        
        # Add race distribution if available
        if 'race' in analysis:
            for race, count in analysis['race'].items():
                summary[f'Race_{race}'] = count
        
        # Add age group distribution if available
        if 'age_group' in analysis:
            for age, count in analysis['age_group'].items():
                summary[f'Age_{age}'] = count
        
        # Add education distribution
        if 'education' in analysis:
            for edu, count in analysis['education'].items():
                summary[f'Education_{edu}'] = count
        
        # Add marital distribution
        if 'marital' in analysis:
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
    
    # Return formatted results
    return {
        'summary': summary_df,
        'detailed_assignments': detailed_output,
        'jury_analysis': jury_analysis,
        'similarity_metrics': assignment_results.get('similarity_metrics', {}),
        'solution_quality': {
            'status': assignment_results['solution_status'],
            'objective_value': assignment_results['objective_value'],
            'deviations': assignment_results['deviations'],
            'tier_weights': assignment_results.get('tier_weights', {}),
            'optimization_method': 'Simultaneous Jury Optimization'
        }
    }


if __name__ == "__main__":
    # Example usage (for testing)
    from analysis.data_processing import process_juror_data
    
    file_path = r"C:\Users\NicholasWilson\OneDrive - Trial Behavior Consulting\County_Split.xlsx"
    num_juries = 3
    jury_size = 8
    
    try:
        # Process the data
        data_dict = process_juror_data(file_path, num_juries, jury_size)
        
        # Define tier weights
        tier_weights = {
            'tier1_pd_balance': 1000.0,
            'tier2_granular_leaning': 100.0,
            'tier2_gender': 90.0,
            'tier3_race': 5.0,
            'tier3_age': 3.0,
            'tier3_education': 2.0,
            'tier3_marital': 1.0
        }
        
        # Run simultaneous optimization
        results = optimize_jury_assignment(data_dict, tier_weights)
        
        # Print summary
        print("\nJury Assignment Summary:")
        print(results['summary'])
        
        # Print solution quality
        print("\nSolution Quality:")
        solution_quality = results['solution_quality']
        print(f"Status: {solution_quality['status']}")
        print(f"Method: {solution_quality['optimization_method']}")
        print(f"Objective Value: {solution_quality['objective_value']:.2f}")
        
        # Print similarity metrics
        if 'similarity_metrics' in results:
            print("\nSimilarity Metrics:")
            print("Pairwise Differences:")
            for key, val in results['similarity_metrics']['pairwise_differences'].items():
                print(f"  {key}: {val}")
        
    except Exception as e:
        import traceback
        print(f"Error in simultaneous optimization process: {str(e)}")
        print(traceback.format_exc())