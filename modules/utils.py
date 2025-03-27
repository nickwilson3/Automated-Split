# utils.py
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import seaborn as sns
import os
from datetime import datetime
import numpy as np


def visualize_jury_demographics(results, output_dir=None):
    """
    Create visualizations of jury demographics.
    
    Parameters:
    results (dict): Formatted results from the optimization process
    output_dir (str, optional): Directory to save the visualizations
    
    Returns:
    dict: Dictionary of created figure objects
    """
    figures = {}
    
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get jury analysis data
    jury_analysis = results['jury_analysis']
    
    # 1. Create P/D leaning distribution visualization
    fig_leaning, ax_leaning = plt.subplots(figsize=(10, 6))
    
    leaning_data = []
    for jury_num, analysis in jury_analysis.items():
        # Handle both numeric and alphabetic jury IDs
        if isinstance(jury_num, str) and jury_num.isalpha():
            jury_label = f'Jury {jury_num}'  # Already a letter like 'A'
        else:
            try:
                jury_label = f'Jury {chr(64 + int(jury_num))}'
            except (ValueError, TypeError):
                jury_label = f'Jury {jury_num}'  # Fall back to original value
                
        # Match the exact case of keys used in optimization.py
        if 'leaning' in analysis:
            for leaning, count in analysis['leaning'].items():
                leaning_data.append({
                    'Jury': jury_label,
                    'Leaning': leaning,
                    'Count': count
                })
    
    if leaning_data:
        leaning_df = pd.DataFrame(leaning_data)
        
        leaning_order = ['P+', 'P', 'D', 'D+']

        # Create the grouped bar chart
        sns.barplot(x='Jury', y='Count', hue='Leaning', hue_order=leaning_order, data=leaning_df, ax=ax_leaning)
        ax_leaning.set_title('Juror Leaning Distribution by Jury')
        ax_leaning.set_ylabel('Number of Jurors')
        ax_leaning.legend(title='Leaning')
        
        figures['leaning'] = fig_leaning
        
        # Save the figure if output directory is provided
        if output_dir:
            fig_path = os.path.join(output_dir, 'leaning_distribution.png')
            fig_leaning.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig_leaning)  # Close figure to free memory
    
    # Create gender distribution visualization
    fig_gender, ax_gender = plt.subplots(figsize=(10, 6))

    gender_data = []
    for jury_num, analysis in jury_analysis.items():
        # Handle both numeric and alphabetic jury IDs
        if isinstance(jury_num, str) and jury_num.isalpha():
            jury_label = f'Jury {jury_num}'  # Already a letter like 'A'
        else:
            try:
                jury_label = f'Jury {chr(64 + int(jury_num))}'
            except (ValueError, TypeError):
                jury_label = f'Jury {jury_num}'  # Fall back to original value
                
        if 'gender' in analysis:
            for gender, count in analysis['gender'].items():
                gender_data.append({
                    'Jury': jury_label,
                    'Gender': gender,
                    'Count': count
                })

    if gender_data:
        gender_df = pd.DataFrame(gender_data)
        
        # Create the grouped bar chart
        sns.barplot(x='Jury', y='Count', hue='Gender', data=gender_df, ax=ax_gender)
        ax_gender.set_title('Juror Gender Distribution by Jury')
        ax_gender.set_ylabel('Number of Jurors')
        ax_gender.legend(title='Gender')
        
        figures['gender'] = fig_gender
        
        # Save the figure if output directory is provided
        if output_dir:
            fig_path = os.path.join(output_dir, 'gender_distribution.png')
            fig_gender.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig_gender)  # Close figure to free memory

    # 2. Create race distribution visualization
    fig_race, ax_race = plt.subplots(figsize=(12, 6))
    
    race_data = []
    for jury_num, analysis in jury_analysis.items():
        # Handle both numeric and alphabetic jury IDs
        if isinstance(jury_num, str) and jury_num.isalpha():
            jury_label = f'Jury {jury_num}'  # Already a letter like 'A'
        else:
            try:
                jury_label = f'Jury {chr(64 + int(jury_num))}'
            except (ValueError, TypeError):
                jury_label = f'Jury {jury_num}'  # Fall back to original value
                
        if 'race' in analysis:
            for race, count in analysis['race'].items():
                race_data.append({
                    'Jury': jury_label,
                    'Race': race,
                    'Count': count
                })
    
    if race_data:
        race_df = pd.DataFrame(race_data)
        
        # Create the grouped bar chart
        sns.barplot(x='Jury', y='Count', hue='Race', data=race_df, ax=ax_race)
        ax_race.set_title('Juror Race Distribution by Jury')
        ax_race.set_ylabel('Number of Jurors')
        ax_race.legend(title='Race')
        
        figures['race'] = fig_race
        
        # Save the figure if output directory is provided
        if output_dir:
            fig_path = os.path.join(output_dir, 'race_distribution.png')
            fig_race.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig_race)  # Close figure to free memory
    
    # 3. Create age group distribution visualization
    fig_age, ax_age = plt.subplots(figsize=(12, 6))
    
    age_data = []
    for jury_num, analysis in jury_analysis.items():
        # Handle both numeric and alphabetic jury IDs
        if isinstance(jury_num, str) and jury_num.isalpha():
            jury_label = f'Jury {jury_num}'  # Already a letter like 'A'
        else:
            try:
                jury_label = f'Jury {chr(64 + int(jury_num))}'
            except (ValueError, TypeError):
                jury_label = f'Jury {jury_num}'  # Fall back to original value
                
        if 'age_group' in analysis:
            for age, count in analysis['age_group'].items():
                age_data.append({
                    'Jury': jury_label,
                    'Age Group': age,
                    'Count': count
                })
    
    if age_data:
        age_df = pd.DataFrame(age_data)
        age_group_order = ['<30', '30-39', '40-49', '50-59', '60+']
        # Create the grouped bar chart
        sns.barplot(x='Jury', y='Count', hue='Age Group', hue_order=age_group_order, data=age_df, ax=ax_age)
        ax_age.set_title('Juror Age Distribution by Jury')
        ax_age.set_ylabel('Number of Jurors')
        ax_age.legend(title='Age Group')
        
        figures['age'] = fig_age
        
        # Save the figure if output directory is provided
        if output_dir:
            fig_path = os.path.join(output_dir, 'age_distribution.png')
            fig_age.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig_age)  # Close figure to free memory
    
    # 4. Create education distribution visualization
    fig_education, ax_education = plt.subplots(figsize=(12, 6))
    
    education_data = []
    for jury_num, analysis in jury_analysis.items():
        # Handle both numeric and alphabetic jury IDs
        if isinstance(jury_num, str) and jury_num.isalpha():
            jury_label = f'Jury {jury_num}'  # Already a letter like 'A'
        else:
            try:
                jury_label = f'Jury {chr(64 + int(jury_num))}'
            except (ValueError, TypeError):
                jury_label = f'Jury {jury_num}'  # Fall back to original value
                
        if 'education' in analysis:
            for education, count in analysis['education'].items():
                education_data.append({
                    'Jury': jury_label,
                    'Education': education,
                    'Count': count
                })
    
    if education_data:
        education_df = pd.DataFrame(education_data)
        
        # Create the grouped bar chart
        sns.barplot(x='Jury', y='Count', hue='Education', data=education_df, ax=ax_education)
        ax_education.set_title('Juror Education Distribution by Jury')
        ax_education.set_ylabel('Number of Jurors')
        ax_education.legend(title='Education')
        
        figures['education'] = fig_education
        
        # Save the figure if output directory is provided
        if output_dir:
            fig_path = os.path.join(output_dir, 'education_distribution.png')
            fig_education.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig_education)  # Close figure to free memory
    
    # 5. Create marital status distribution visualization
    fig_marital, ax_marital = plt.subplots(figsize=(12, 6))
    
    marital_data = []
    for jury_num, analysis in jury_analysis.items():
        if isinstance(jury_num, str) and jury_num.isalpha():
            jury_label = f'Jury {jury_num}'
        else:
            try:
                jury_label = f'Jury {chr(64 + int(jury_num))}'
            except (ValueError, TypeError):
                jury_label = f'Jury {jury_num}'
                
        if 'marital' in analysis:
            for marital, count in analysis['marital'].items():
                marital_data.append({
                    'Jury': jury_label,
                    'Marital Status': marital,
                    'Count': count
                })
    
    if marital_data:
        marital_df = pd.DataFrame(marital_data)
        
        # Create the grouped bar chart
        sns.barplot(x='Jury', y='Count', hue='Marital Status', data=marital_df, ax=ax_marital)
        ax_marital.set_title('Juror Marital Status Distribution by Jury')
        ax_marital.set_ylabel('Number of Jurors')
        ax_marital.legend(title='Marital Status')
        
        figures['marital'] = fig_marital
        
        # Save the figure if output directory is provided
        if output_dir:
            fig_path = os.path.join(output_dir, 'marital_distribution.png')
            fig_marital.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig_marital)  # Close figure to free memory
    
    # 6. Create summary heatmap showing balance across all demographics
    # Prepare data for the heatmap
    heatmap_data = {}
    # Use the exact keys used in optimization.py's jury_analysis
    demographic_vars = ['leaning', 'race', 'age_group', 'education', 'marital']
    
    for jury_num in jury_analysis:
        # Handle both numeric and alphabetic jury IDs
        if isinstance(jury_num, str) and jury_num.isalpha():
            jury_label = f'Jury {jury_num}'  # Already a letter like 'A'
        else:
            try:
                jury_label = f'Jury {chr(64 + int(jury_num))}'
            except (ValueError, TypeError):
                jury_label = f'Jury {jury_num}'  # Fall back to original value
                
        heatmap_data[jury_label] = {}
        
        # Calculate P-D balance
        if 'leaning' in jury_analysis[jury_num]:
            leaning_counts = jury_analysis[jury_num]['leaning']
            p_count = sum(count for leaning, count in leaning_counts.items() if leaning in ['P', 'P+'])
            d_count = sum(count for leaning, count in leaning_counts.items() if leaning in ['D', 'D+'])
            
            # Calculate balance as absolute deviation from 50-50
            jury_size = p_count + d_count
            if jury_size > 0:
                p_percentage = p_count / jury_size
                heatmap_data[jury_label]['P-D Balance'] = 1 - abs(p_percentage - 0.5) * 2  # 1 = perfect balance, 0 = worst balance
    
    # Create the heatmap if we have data
    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data).T.fillna(0)
        
        if not heatmap_df.empty and heatmap_df.shape[1] > 0:
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, len(heatmap_df) * 0.8))
            sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, ax=ax_heatmap)
            ax_heatmap.set_title('Jury Balance Metrics (1 = Perfect Balance)')
            
            figures['balance_heatmap'] = fig_heatmap
            
            # Save the figure if output directory is provided
            if output_dir:
                fig_path = os.path.join(output_dir, 'balance_heatmap.png')
                fig_heatmap.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close(fig_heatmap)  # Close figure to free memory
    
    return figures


# In utils.py
def export_results_to_excel(results, output_path=None):
    """
    Export the optimization results to an Excel file.
    
    Parameters:
    results (dict): Formatted results from the optimization process
    output_path (str, optional): Path to save the Excel file
    
    Returns:
    str: Path to the saved Excel file
    """
    # Create a default output path if none is provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"jury_assignments_{timestamp}.xlsx"
    
    # Debug: print the summary DataFrame columns before writing to Excel
    if 'summary' in results:
        print("Summary DataFrame columns before Excel export:")
        print(results['summary'].columns.tolist())
    
    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Export summary to a sheet
        if 'summary' in results:
            # Make sure we're exporting the entire DataFrame
            results['summary'].to_excel(writer, sheet_name='Summary', index=False)
            
            # Debug: Confirm what was written
            print(f"Wrote {len(results['summary'])} rows and {len(results['summary'].columns)} columns to Summary sheet")
        
        # Rest of the function remains the same...
        # Export detailed assignments to a sheet
        if 'detailed_assignments' in results:
            results['detailed_assignments'].to_excel(writer, sheet_name='Assignments', index=False)
        
        # Export solution quality metrics to a sheet
        if 'solution_quality' in results:
            solution_quality = results['solution_quality']
            
            # Convert solution quality to DataFrame
            sq_data = {
                'Metric': ['Status', 'Objective Value'],
                'Value': [solution_quality['status'], solution_quality['objective_value']]
            }
            
            # Add deviations
            if 'deviations' in solution_quality:
                for var, dev in solution_quality['deviations'].items():
                    sq_data['Metric'].append(f"{var} Deviation")
                    sq_data['Value'].append(dev)
            
            sq_df = pd.DataFrame(sq_data)
            sq_df.to_excel(writer, sheet_name='Solution_Quality', index=False)
        
        # Export balance metrics to a sheet
        if 'balance_metrics' in results:
            balance_metrics = results['balance_metrics']
            
            # Convert balance metrics to DataFrames
            for metric_name, metrics in balance_metrics.items():
                bm_data = {
                    'Metric': list(metrics.keys()),
                    'Value': list(metrics.values())
                }
                
                bm_df = pd.DataFrame(bm_data)
                bm_df.to_excel(writer, sheet_name=f'{metric_name.capitalize()}_Balance', index=False)
        
        # Export individual jury analysis to separate sheets
        if 'jury_analysis' in results:
            jury_analysis = results['jury_analysis']
            
            for jury_num, analysis in jury_analysis.items():
                # Handle both numeric and alphabetic jury IDs
                if isinstance(jury_num, str) and jury_num.isalpha():
                    jury_label = jury_num  # Already a letter like 'A'
                else:
                    try:
                        jury_label = chr(64 + int(jury_num))
                    except (ValueError, TypeError):
                        jury_label = str(jury_num)
                
                # Create a DataFrame for this jury's jurors
                if 'jurors' in analysis:
                    jury_df = pd.DataFrame(analysis['jurors'])
                    jury_df.to_excel(writer, sheet_name=f'Jury_{jury_label}_Details', index=False)
    
    print(f"Results exported to {output_path}")
    return output_path


def create_html_report(results, output_path=None, include_visualizations=True):
    """
    Create an HTML report of the optimization results with embedded base64 images.
    
    Parameters:
    results (dict): Formatted results from the optimization process
    output_path (str, optional): Path to save the HTML file
    include_visualizations (bool): Whether to include visualizations in the report
    
    Returns:
    str: Path to the saved HTML file
    """
    import base64
    import matplotlib.pyplot as plt
    import seaborn as sns
    from io import BytesIO
    
    # Create a default output path if none is provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"jury_report_{timestamp}.html"
    
    # Dictionary to store base64-encoded images
    embedded_images = {}
    
    # Create and embed visualizations if requested
    if include_visualizations:
        # Get jury analysis data
        jury_analysis = results['jury_analysis']
        
        # 1. Leaning distribution
        leaning_data = []
        for jury_num, analysis in jury_analysis.items():
            # Get jury label
            if isinstance(jury_num, str) and jury_num.isalpha():
                jury_label = f'Jury {jury_num}'
            else:
                try:
                    jury_label = f'Jury {chr(64 + int(jury_num))}'
                except:
                    jury_label = f'Jury {jury_num}'
                    
            if 'leaning' in analysis:
                for leaning, count in analysis['leaning'].items():
                    leaning_data.append({
                        'Jury': jury_label,
                        'Leaning': leaning,
                        'Count': count
                    })
        
        if leaning_data:
            # Create the visualization in memory
            fig, ax = plt.subplots(figsize=(10, 6))
            leaning_df = pd.DataFrame(leaning_data)
            leaning_order = ['P+', 'P', 'D', 'D+']
            sns.barplot(x='Jury', y='Count', hue='Leaning', hue_order=leaning_order, data=leaning_df, ax=ax)
            ax.set_title('Juror Leaning Distribution by Jury')
            ax.set_ylabel('Number of Jurors')
            ax.legend(title='Leaning')
            
            # Save to bytes buffer
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            
            # Encode as base64
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            embedded_images['leaning'] = img_str
            
            plt.close(fig)  # Close figure to free memory
        
        # 2. Gender distribution
        gender_data = []
        for jury_num, analysis in jury_analysis.items():
            if isinstance(jury_num, str) and jury_num.isalpha():
                jury_label = f'Jury {jury_num}'
            else:
                try:
                    jury_label = f'Jury {chr(64 + int(jury_num))}'
                except:
                    jury_label = f'Jury {jury_num}'
                    
            if 'gender' in analysis:
                for gender, count in analysis['gender'].items():
                    gender_data.append({
                        'Jury': jury_label,
                        'Gender': gender,
                        'Count': count
                    })
        
        if gender_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            gender_df = pd.DataFrame(gender_data)
            sns.barplot(x='Jury', y='Count', hue='Gender', data=gender_df, ax=ax)
            ax.set_title('Juror Gender Distribution by Jury')
            ax.set_ylabel('Number of Jurors')
            ax.legend(title='Gender')
            
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            embedded_images['gender'] = img_str
            
            plt.close(fig)
        
        # 3. Race distribution
        race_data = []
        for jury_num, analysis in jury_analysis.items():
            if isinstance(jury_num, str) and jury_num.isalpha():
                jury_label = f'Jury {jury_num}'
            else:
                try:
                    jury_label = f'Jury {chr(64 + int(jury_num))}'
                except:
                    jury_label = f'Jury {jury_num}'
                    
            if 'race' in analysis:
                for race, count in analysis['race'].items():
                    race_data.append({
                        'Jury': jury_label,
                        'Race': race,
                        'Count': count
                    })
        
        if race_data:
            fig, ax = plt.subplots(figsize=(12, 6))
            race_df = pd.DataFrame(race_data)
            sns.barplot(x='Jury', y='Count', hue='Race', data=race_df, ax=ax)
            ax.set_title('Juror Race Distribution by Jury')
            ax.set_ylabel('Number of Jurors')
            ax.legend(title='Race')
            
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            embedded_images['race'] = img_str
            
            plt.close(fig)
        
        # 4. Age group distribution
        age_data = []
        for jury_num, analysis in jury_analysis.items():
            if isinstance(jury_num, str) and jury_num.isalpha():
                jury_label = f'Jury {jury_num}'
            else:
                try:
                    jury_label = f'Jury {chr(64 + int(jury_num))}'
                except:
                    jury_label = f'Jury {jury_num}'
                    
            if 'age_group' in analysis:
                for age, count in analysis['age_group'].items():
                    age_data.append({
                        'Jury': jury_label,
                        'Age Group': age,
                        'Count': count
                    })
        
        if age_data:
            fig, ax = plt.subplots(figsize=(12, 6))
            age_df = pd.DataFrame(age_data)
            age_group_order = ['<30', '30-39', '40-49', '50-59', '60+']
            sns.barplot(x='Jury', y='Count', hue='Age Group', hue_order=age_group_order, data=age_df, ax=ax)
            ax.set_title('Juror Age Distribution by Jury')
            ax.set_ylabel('Number of Jurors')
            ax.legend(title='Age Group')
            
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            embedded_images['age'] = img_str
            
            plt.close(fig)
        
        # 5. Education distribution
        education_data = []
        for jury_num, analysis in jury_analysis.items():
            if isinstance(jury_num, str) and jury_num.isalpha():
                jury_label = f'Jury {jury_num}'
            else:
                try:
                    jury_label = f'Jury {chr(64 + int(jury_num))}'
                except:
                    jury_label = f'Jury {jury_num}'
                    
            if 'education' in analysis:
                for edu, count in analysis['education'].items():
                    education_data.append({
                        'Jury': jury_label,
                        'Education': edu,
                        'Count': count
                    })
        
        if education_data:
            fig, ax = plt.subplots(figsize=(12, 6))
            edu_df = pd.DataFrame(education_data)
            sns.barplot(x='Jury', y='Count', hue='Education', data=edu_df, ax=ax)
            ax.set_title('Juror Education Distribution by Jury')
            ax.set_ylabel('Number of Jurors')
            ax.legend(title='Education')
            
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            embedded_images['education'] = img_str
            
            plt.close(fig)
        
        # 6. Marital status distribution
        marital_data = []
        for jury_num, analysis in jury_analysis.items():
            if isinstance(jury_num, str) and jury_num.isalpha():
                jury_label = f'Jury {jury_num}'
            else:
                try:
                    jury_label = f'Jury {chr(64 + int(jury_num))}'
                except:
                    jury_label = f'Jury {jury_num}'
                    
            if 'marital' in analysis:
                for status, count in analysis['marital'].items():
                    marital_data.append({
                        'Jury': jury_label,
                        'Marital Status': status,
                        'Count': count
                    })
        
        if marital_data:
            fig, ax = plt.subplots(figsize=(12, 6))
            marital_df = pd.DataFrame(marital_data)
            sns.barplot(x='Jury', y='Count', hue='Marital Status', data=marital_df, ax=ax)
            ax.set_title('Juror Marital Status Distribution by Jury')
            ax.set_ylabel('Number of Jurors')
            ax.legend(title='Marital Status')
            
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            embedded_images['marital'] = img_str
            
            plt.close(fig)
        
        # 7. P-D Balance heatmap
        heatmap_data = {}
        for jury_num in jury_analysis:
            if isinstance(jury_num, str) and jury_num.isalpha():
                jury_label = f'Jury {jury_num}'
            else:
                try:
                    jury_label = f'Jury {chr(64 + int(jury_num))}'
                except:
                    jury_label = f'Jury {jury_num}'
                    
            heatmap_data[jury_label] = {}
            
            # Calculate P-D balance
            if 'leaning' in jury_analysis[jury_num]:
                leaning_counts = jury_analysis[jury_num]['leaning']
                p_count = sum(count for leaning, count in leaning_counts.items() if leaning in ['P', 'P+'])
                d_count = sum(count for leaning, count in leaning_counts.items() if leaning in ['D', 'D+'])
                
                # Calculate balance as absolute deviation from 50-50
                jury_size = p_count + d_count
                if jury_size > 0:
                    p_percentage = p_count / jury_size
                    heatmap_data[jury_label]['P-D Balance'] = 1 - abs(p_percentage - 0.5) * 2  # 1 = perfect balance, 0 = worst balance
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data).T.fillna(0)
            
            if not heatmap_df.empty and heatmap_df.shape[1] > 0:
                fig, ax = plt.subplots(figsize=(10, len(heatmap_df) * 0.8))
                sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
                ax.set_title('Jury Balance Metrics (1 = Perfect Balance)')
                
                buffer = BytesIO()
                fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                
                img_str = base64.b64encode(buffer.read()).decode('utf-8')
                embedded_images['balance'] = img_str
                
                plt.close(fig)
    
    # Generate HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jury Selection Optimization Report</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
            h1, h2, h3 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .jury-box { border: 1px solid #ccc; padding: 10px; margin: 10px 0; background-color: #f9f9f9; }
            .jury-title { font-weight: bold; font-size: 1.2em; margin-bottom: 10px; }
            .section { margin-bottom: 30px; }
            .metric { font-weight: bold; }
            .visualization { margin: 20px 0; text-align: center; }
        </style>
    </head>
    <body>
        <h1>Jury Selection Optimization Report</h1>
        <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        
        <div class="section">
            <h2>Solution Summary</h2>
    """
    
    # Add solution quality information
    if 'solution_quality' in results:
        solution_quality = results['solution_quality']
        html_content += f"""
            <p><span class="metric">Solution Status:</span> {solution_quality['status']}</p>
            <p><span class="metric">Objective Value:</span> {solution_quality['objective_value']}</p>
        """
        
        # Add deviations if available
        if 'deviations' in solution_quality:
            html_content += "<h3>Demographic Balance Deviations</h3>"
            html_content += "<table>"
            html_content += "<tr><th>Demographic Variable</th><th>Total Deviation</th></tr>"
            
            for var, dev in solution_quality['deviations'].items():
                html_content += f"<tr><td>{var}</td><td>{dev}</td></tr>"
            
            html_content += "</table>"
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>Jury Summary</h2>
    """
    
    # Add summary table if available
    if 'summary' in results:
        summary_df = results['summary']
        html_content += summary_df.to_html(index=False)
    
    # Include visualizations
    if embedded_images:
        html_content += """
        <div class="section">
            <h2>Demographic Visualizations</h2>
        """
        
        # Add each visualization using base64-encoded images
        visualization_titles = {
            'leaning': 'Leaning Distribution',
            'gender': 'Gender Distribution',
            'race': 'Race Distribution',
            'age': 'Age Distribution',
            'education': 'Education Distribution',
            'marital': 'Marital Status Distribution',
            'balance': 'Balance Metrics'
        }
        
        for fig_name, img_str in embedded_images.items():
            title = visualization_titles.get(fig_name, fig_name.replace('_', ' ').title())
            html_content += f"""
            <div class="visualization">
                <h3>{title}</h3>
                <img src="data:image/png;base64,{img_str}" alt="{fig_name} visualization" style="max-width: 100%;">
            </div>
            """
        
        html_content += "</div>"
    
    # Add detailed jury information
    html_content += """
        <div class="section">
            <h2>Detailed Jury Assignments</h2>
    """
    
    if 'jury_analysis' in results:
        jury_analysis = results['jury_analysis']
        
        for jury_num, analysis in jury_analysis.items():
            # Handle both numeric and alphabetic jury IDs
            if isinstance(jury_num, str) and jury_num.isalpha():
                jury_label = jury_num  # Already a letter like 'A'
            else:
                try:
                    jury_label = chr(64 + int(jury_num))
                except (ValueError, TypeError):
                    jury_label = str(jury_num)
            
            html_content += f"""
            <div class="jury-box">
                <div class="jury-title">Jury {jury_label}</div>
            """
            
            # Add demographic breakdowns
            for demo_key, demo_name in [
                ('leaning', 'Leaning'),
                ('gender', 'Gender'),
                ('race', 'Race'), 
                ('age_group', 'Age Group'), 
                ('education', 'Education'),
                ('marital', 'Marital Status')
            ]:
                if demo_key in analysis and analysis[demo_key]:
                    html_content += f"<p><strong>{demo_name} Breakdown:</strong> "
                    items = [f"{k}: {v}" for k, v in analysis[demo_key].items()]
                    html_content += ", ".join(items)
                    html_content += "</p>"
            
            # Add table of jurors
            if 'jurors' in analysis and analysis['jurors']:
                html_content += "<h4>Jurors</h4>"
                juror_df = pd.DataFrame(analysis['jurors'])
                
                # Select relevant columns, ensure Education is included
                display_columns = ['Name', 'Final_Leaning', 'Gender', 'Race', 'Age', 'AgeGroup', 
                                  'Education', 'Marital']
                display_columns = [col for col in display_columns if col in juror_df.columns]
                
                html_content += juror_df[display_columns].to_html(index=False)
            
            html_content += "</div>"
    
    # Add education balance metrics section
    if 'balance_metrics' in results and 'education' in results['balance_metrics']:
        html_content += """
        <div class="section">
            <h2>Education Balance Analysis</h2>
            <table>
                <tr><th>Education Level</th><th>Count</th><th>Percentage</th></tr>
        """
        
        education_metrics = results['balance_metrics']['education']
        
        for edu_level, count in education_metrics['counts'].items():
            percentage = education_metrics['percentages'].get(edu_level, 0)
            html_content += f"<tr><td>{edu_level}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        html_content += """
            </table>
        </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {output_path}")
    return output_path



def analyze_optimization_results(results, data_dict):
    """
    Analyze the optimization results to provide insights.
    
    Parameters:
    results (dict): Formatted results from the optimization process
    data_dict (dict): Dictionary containing processed juror data
    
    Returns:
    dict: Analysis metrics and insights
    """
    analysis = {
        'metrics': {},
        'insights': []
    }
    
    # Existing analysis code...
    
    # Add education distribution analysis
    if 'balance_metrics' in results and 'education' in results['balance_metrics']:
        education = results['balance_metrics']['education']
        
        # Store education metrics
        analysis['metrics']['education_distribution'] = education['counts']
        analysis['metrics']['education_percentages'] = education['percentages']
        
        # Find most common education level
        if education['counts']:
            most_common_edu = max(education['counts'], key=education['counts'].get)
            most_common_pct = education['percentages'].get(most_common_edu, 0)
            
            analysis['metrics']['most_common_education'] = most_common_edu
            analysis['metrics']['most_common_education_pct'] = most_common_pct
            
            # Add insight about education distribution
            analysis['insights'].append(f"The most common education level is '{most_common_edu}' ({most_common_pct:.1f}% of jurors).")
        
        # Check education distribution across juries
        if 'jury_analysis' in results:
            jury_analysis = results['jury_analysis']
            
            # Calculate education distribution variance across juries
            education_variance = {}
            
            for jury_num, jury_data in jury_analysis.items():
                if 'education' in jury_data:
                    for edu, count in jury_data['education'].items():
                        if edu not in education_variance:
                            education_variance[edu] = []
                        education_variance[edu].append(count)
            
            # Calculate coefficient of variation for each education level
            high_variance_edu = []
            for edu, counts in education_variance.items():
                if len(counts) > 1:  # Need at least 2 juries to calculate variance
                    mean = sum(counts) / len(counts)
                    if mean > 0:
                        variance = sum((x - mean) ** 2 for x in counts) / len(counts)
                        std_dev = variance ** 0.5
                        cv = std_dev / mean
                        
                        if cv > 0.5:  # High variance threshold
                            high_variance_edu.append((edu, cv))
            
            if high_variance_edu:
                edu_list = ", ".join([f"'{edu}' (CV={cv:.2f})" for edu, cv in high_variance_edu])
                analysis['insights'].append(f"There is significant variation in the distribution of education levels across juries for: {edu_list}")
    
    return analysis
# Add these functions to utils.py

def export_assignments_for_editing(results, output_path=None):
    """
    Export jury assignments to an Excel file for manual editing.
    
    Parameters:
    results (dict): Formatted results from the optimization process
    output_path (str, optional): Path to save the Excel file
    
    Returns:
    str: Path to the saved Excel file
    """
    # Create a default output path if none is provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"jury_assignments_for_editing_{timestamp}.xlsx"
    
    # Get the detailed assignments DataFrame
    if 'detailed_assignments' in results:
        # Convert to DataFrame if it's a list
        if isinstance(results['detailed_assignments'], list):
            assignments_df = pd.DataFrame(results['detailed_assignments'])
        else:
            assignments_df = results['detailed_assignments'].copy()
        
        # Add a note at the top of the sheet to guide the user
        # We'll create a new DataFrame for the instruction
        instruction = pd.DataFrame({
            'INSTRUCTIONS': ['Edit the "jury" column to reassign jurors. Save the file when done. DO NOT change the column names or structure.']
        })
        
        # Write to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            instruction.to_excel(writer, sheet_name='Assignments', index=False)
            assignments_df.to_excel(writer, sheet_name='Assignments', startrow=2, index=False)
        
        print(f"Assignment data exported for editing to {output_path}")
        print("Please edit the jury assignments in the file and save it.")
        print("When finished, use the load_edited_assignments function with the same file path.")
        
        return output_path
    else:
        print("Error: No assignment data found in results.")
        return None


# Replace in utils.py
def load_edited_assignments(assignments_file_path, original_results):
    try:
        # Load the edited assignments
        edited_df = pd.read_excel(assignments_file_path, sheet_name='Assignments', skiprows=2)
        
        # Make a copy of the original results
        updated_results = original_results.copy()
        
        # Update the detailed assignments
        updated_results['detailed_assignments'] = edited_df
        
        # Recalculate jury analysis based on edited assignments
        jury_analysis = {}
        
        # Get unique jury numbers
        jury_indices = edited_df['jury'].unique()
        
        for j in jury_indices:
            # Filter for this jury
            jury_j = edited_df[edited_df['jury'] == j]
            
            # Count demographics for this jury
            analysis = {
                'size': len(jury_j),
                'leaning': jury_j['Final_Leaning'].value_counts().to_dict() if 'Final_Leaning' in jury_j.columns else {},
                'gender': jury_j['Gender'].value_counts().to_dict() if 'Gender' in jury_j.columns else {},
                'race': jury_j['Race'].value_counts().to_dict() if 'Race' in jury_j.columns else {},
                'age_group': jury_j['AgeGroup'].value_counts().to_dict() if 'AgeGroup' in jury_j.columns else {},
                'education': jury_j['Education'].value_counts().to_dict() if 'Education' in jury_j.columns else {},
                'marital': jury_j['Marital'].value_counts().to_dict() if 'Marital' in jury_j.columns else {},
                'jurors': jury_j.to_dict('records')
            }
            
            jury_analysis[j] = analysis
        
        # Update the jury analysis in the results
        updated_results['jury_analysis'] = jury_analysis
        
        # Update summary DataFrame
        jury_summaries = []
        
        for jury_num, analysis in jury_analysis.items():
            # Count P and D leanings
            p_count = sum(count for leaning, count in analysis['leaning'].items() 
                         if leaning in ['P', 'P+'])
            d_count = sum(count for leaning, count in analysis['leaning'].items() 
                         if leaning in ['D', 'D+'])
            
            summary = {
                'Jury': chr(64 + int(jury_num)),
                'Size': analysis['size'],
                'P_Leaning': p_count,
                'D_Leaning': d_count,
                'P_D_Ratio': f"{p_count}:{d_count}"
            }
            
            # Add race distribution if available
            if 'race' in analysis and analysis['race']:
                for race, count in analysis['race'].items():
                    summary[f'Race_{race}'] = count
            
            if 'gender' in analysis and analysis['gender']:
                for gender, count in analysis['gender'].items():
                    summary[{gender}] = count
            
            # Add age group distribution if available
            if 'age_group' in analysis and analysis['age_group']:
                for age, count in analysis['age_group'].items():
                    summary[f'Age_{age}'] = count
            
            # Add education distribution if available - this is the key addition
            if 'education' in analysis and analysis['education']:
                for edu, count in analysis['education'].items():
                    summary[f'Education_{edu}'] = count
            
            jury_summaries.append(summary)
        
        if jury_summaries:
            updated_results['summary'] = pd.DataFrame(jury_summaries)
        
        # Store education columns explicitly to ensure they're preserved
        if 'summary' in updated_results:
            education_columns = [col for col in updated_results['summary'].columns if col.startswith('Education_')]
            updated_results['education_columns'] = education_columns
        
        # Recalculate balance metrics
        balance_metrics = {}
        
        # Leaning balance
        if 'Final_Leaning' in edited_df.columns:
            overall_p = edited_df['Final_Leaning'].isin(['P', 'P+']).sum()
            overall_d = edited_df['Final_Leaning'].isin(['D', 'D+']).sum()
            
            leaning_balance = {
                'overall_p': overall_p,
                'overall_d': overall_d,
                'p_percentage': round(overall_p / len(edited_df) * 100, 2)
            }
            
            balance_metrics['leaning'] = leaning_balance
        
        updated_results['balance_metrics'] = balance_metrics
        
        # Update solution quality to indicate manual editing
        if 'solution_quality' in updated_results:
            updated_results['solution_quality']['status'] = "Manually Edited"
            updated_results['solution_quality']['notes'] = "Assignments were manually adjusted after optimization"
        
        print(f"Successfully loaded edited assignments from {assignments_file_path}")
        print(f"Updated results reflect manual adjustments")
        
        # Debug - check for education columns
        if 'summary' in updated_results:
            print("Education columns in final summary:")
            edu_cols = [col for col in updated_results['summary'].columns if col.startswith('Education_')]
            print(edu_cols)
        
        return updated_results
    
    except Exception as e:
        print(f"Error loading edited assignments: {str(e)}")
        return original_results

def manual_adjustment_workflow(results, output_path=None, wait_for_edit=True):
    """
    Run the complete manual adjustment workflow, including exporting,
    waiting for user edits, and reloading the assignments.
    
    Parameters:
    results (dict): Original results from the optimization process
    output_path (str, optional): Path to save the Excel file
    wait_for_edit (bool): If True, script will pause and wait for user to edit file
    
    Returns:
    dict: Updated results after manual adjustments
    """
    # Export the assignments for editing
    edit_file_path = export_assignments_for_editing(results, output_path)
    
    if edit_file_path is None:
        print("Failed to export assignments for editing.")
        return results
    
    if wait_for_edit:
        # Pause execution and wait for user to edit the file
        input(f"\nPlease edit the assignments in '{edit_file_path}' and save the file.\nPress Enter when you are done editing...")
    else:
        # Just show a message but don't wait
        print(f"\nPlease edit the assignments in '{edit_file_path}' and save the file.")
        print("Then run the load_edited_assignments function with the same file path.")
        return results, edit_file_path
    
    # Load the edited assignments
    updated_results = load_edited_assignments(edit_file_path, results)
    
    return updated_results