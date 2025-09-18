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
    
    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Export summary to a sheet
        if 'summary' in results:
            results['summary'].to_excel(writer, sheet_name='Summary', index=False)
        
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
            
            # Add hierarchical balance info
            if 'hierarchical_balance' in solution_quality:
                for key, value in solution_quality['hierarchical_balance'].items():
                    sq_data['Metric'].append(f"Hierarchical Balance - {key}")
                    sq_data['Value'].append(value)
            
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
            .tier-info { background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>Jury Selection Optimization Report</h1>
        <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        
        <div class="section">
            <h2>Hierarchical Optimization Summary</h2>
            <div class="tier-info">
                <h4>Constraint Hierarchy Applied:</h4>
                <p><strong>TIER 1 (INVIOLABLE):</strong> Jury size and basic P/D balance - hard constraints that can never be violated</p>
                <p><strong>TIER 2 (SECONDARY):</strong> Granular P+/P/D/D+ balance and gender balance - soft constraints with high penalty</p>
                <p><strong>TIER 3 (TERTIARY):</strong> Other demographics - weighted optimization with lower penalty</p>
            </div>
    """
    
    # Add solution quality information
    if 'solution_quality' in results:
        solution_quality = results['solution_quality']
        html_content += f"""
            <p><span class="metric">Solution Status:</span> {solution_quality['status']}</p>
            <p><span class="metric">Objective Value:</span> {solution_quality['objective_value']}</p>
        """
        
        # Add hierarchical balance results
        if 'hierarchical_balance' in solution_quality:
            balance = solution_quality['hierarchical_balance']
            html_content += "<h3>Hierarchical Balance Achievement</h3>"
            html_content += "<table>"
            html_content += "<tr><th>Tier</th><th>Constraint Type</th><th>Achieved</th></tr>"
            html_content += f"<tr><td>TIER 1</td><td>Jury Size Correct</td><td>{'✓' if balance.get('tier1_jury_size', False) else '✗'}</td></tr>"
            html_content += f"<tr><td>TIER 1</td><td>Basic P/D Balance</td><td>{'✓' if balance.get('tier1_basic_pd', False) else '✗'}</td></tr>"
            html_content += f"<tr><td>TIER 2</td><td>Granular P+/P/D/D+ Balance</td><td>{'✓' if balance.get('tier2_granular', False) else '✗'}</td></tr>"
            html_content += f"<tr><td>TIER 2</td><td>Gender Balance</td><td>{'✓' if balance.get('tier2_gender', False) else '✗'}</td></tr>"
            html_content += "</table>"
        
        # Add deviations if available
        if 'deviations' in solution_quality:
            html_content += "<h3>Soft Constraint Deviations</h3>"
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
                display_columns = ['Name', '#', 'Final_Leaning', 'Gender', 'Race', 'Age', 'AgeGroup', 
                                  'Education', 'Marital']
                display_columns = [col for col in display_columns if col in juror_df.columns]
                
                html_content += juror_df[display_columns].to_html(index=False)
            
            html_content += "</div>"
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {output_path}")
    return output_path


def analyze_optimization_results(results, data_dict):
    """
    Analyze the optimization results to provide insights with hierarchical focus.
    
    Parameters:
    results (dict): Formatted results from the optimization process
    data_dict (dict): Dictionary containing processed juror data
    
    Returns:
    dict: Analysis metrics and insights
    """
    analysis = {
        'metrics': {},
        'insights': [],
        'hierarchical_assessment': {}
    }
    
    # Assess hierarchical constraint achievement
    if 'solution_quality' in results and 'hierarchical_balance' in results['solution_quality']:
        balance = results['solution_quality']['hierarchical_balance']
        
        analysis['hierarchical_assessment'] = {
            'tier1_success': balance.get('tier1_jury_size', False) and balance.get('tier1_basic_pd', False),
            'tier2_success': balance.get('tier2_granular', False) and balance.get('tier2_gender', False),
            'overall_quality': 'excellent' if balance.get('tier1_jury_size', False) and balance.get('tier1_basic_pd', False) 
                              else 'poor'
        }
        
        # Add insights based on hierarchical success
        if analysis['hierarchical_assessment']['tier1_success']:
            analysis['insights'].append("TIER 1 constraints satisfied: All juries have correct size and basic P/D balance.")
        else:
            analysis['insights'].append("WARNING: TIER 1 constraint violations detected. This should not happen with proper optimization.")
        
        if analysis['hierarchical_assessment']['tier2_success']:
            analysis['insights'].append("TIER 2 optimization successful: Achieved both granular P+/P/D/D+ balance and gender balance.")
        else:
            if not balance.get('tier2_granular', False):
                analysis['insights'].append("TIER 2: Granular P+/P/D/D+ balance not fully achieved due to limited juror availability in specific subcategories.")
            if not balance.get('tier2_gender', False):
                analysis['insights'].append("TIER 2: Gender balance not fully achieved due to available juror pool constraints.")
    
    # Existing analysis code for education and other demographics...
    if 'balance_metrics' in results and 'leaning' in results['balance_metrics']:
        leaning = results['balance_metrics']['leaning']
        
        # Store leaning metrics
        analysis['metrics']['overall_p'] = leaning.get('overall_p', 0)
        analysis['metrics']['overall_d'] = leaning.get('overall_d', 0)
        analysis['metrics']['p_percentage'] = leaning.get('p_percentage', 0)
        
        # Add insight about overall balance
        if leaning.get('tier1_optimal_achieved', False):
            analysis['insights'].append(f"Basic P/D balance optimal: {leaning['overall_p']} P-leaning and {leaning['overall_d']} D-leaning jurors assigned.")
        
        # Granular leaning analysis
        if 'granular_counts' in leaning:
            granular = leaning['granular_counts']
            analysis['metrics']['granular_distribution'] = granular
            
            total_granular = sum(granular.values())
            if total_granular > 0:
                p_plus_pct = round(granular.get('P+', 0) / total_granular * 100, 1)
                p_pct = round(granular.get('P', 0) / total_granular * 100, 1)
                d_pct = round(granular.get('D', 0) / total_granular * 100, 1)
                d_plus_pct = round(granular.get('D+', 0) / total_granular * 100, 1)
                
                analysis['insights'].append(f"Granular leaning distribution: P+ ({p_plus_pct}%), P ({p_pct}%), D ({d_pct}%), D+ ({d_plus_pct}%)")
    
    return analysis


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


def load_edited_assignments(assignments_file_path, original_results):
    """
    Load edited assignments from Excel file and update results.
    
    Parameters:
    assignments_file_path (str): Path to the edited Excel file
    original_results (dict): Original optimization results
    
    Returns:
    dict: Updated results with edited assignments
    """
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
            
            # Handle jury label conversion
            if isinstance(jury_num, str) and jury_num.isalpha():
                jury_label = jury_num
            else:
                try:
                    jury_label = chr(64 + int(jury_num))
                except (ValueError, TypeError):
                    jury_label = str(jury_num)
            
            summary = {
                'Jury': jury_label,
                'Size': analysis['size'],
                'P_Leaning': p_count,
                'D_Leaning': d_count,
                'P_D_Ratio': f"{p_count}:{d_count}",
                'P_Overall': p_count,
                'D_Overall': d_count,
                'P+_Count': analysis['leaning'].get('P+', 0),
                'P_Count': analysis['leaning'].get('P', 0),
                'D_Count': analysis['leaning'].get('D', 0),
                'D+_Count': analysis['leaning'].get('D+', 0)
            }
            
            # Add demographic distributions
            if 'gender' in analysis and analysis['gender']:
                for gender, count in analysis['gender'].items():
                    summary[f'Gender_{gender}'] = count
            
            if 'race' in analysis and analysis['race']:
                for race, count in analysis['race'].items():
                    summary[f'Race_{race}'] = count
            
            if 'age_group' in analysis and analysis['age_group']:
                for age, count in analysis['age_group'].items():
                    summary[f'Age_{age}'] = count
            
            if 'education' in analysis and analysis['education']:
                for edu, count in analysis['education'].items():
                    summary[f'Education_{edu}'] = count
            
            if 'marital' in analysis and analysis['marital']:
                for marital, count in analysis['marital'].items():
                    summary[f'Marital_{marital}'] = count
            
            jury_summaries.append(summary)
        
        if jury_summaries:
            updated_results['summary'] = pd.DataFrame(jury_summaries)
        
        # Recalculate balance metrics
        balance_metrics = {}
        
        # Leaning balance
        if 'Final_Leaning' in edited_df.columns:
            overall_p = edited_df['Final_Leaning'].isin(['P', 'P+']).sum()
            overall_d = edited_df['Final_Leaning'].isin(['D', 'D+']).sum()
            
            # Granular counts
            granular_counts = {
                'P+': (edited_df['Final_Leaning'] == 'P+').sum(),
                'P': (edited_df['Final_Leaning'] == 'P').sum(),
                'D': (edited_df['Final_Leaning'] == 'D').sum(),
                'D+': (edited_df['Final_Leaning'] == 'D+').sum()
            }
            
            leaning_balance = {
                'overall_p': overall_p,
                'overall_d': overall_d,
                'p_percentage': round(overall_p / len(edited_df) * 100, 2),
                'granular_counts': granular_counts,
                'tier1_optimal_achieved': False,  # Manual edits may not maintain optimality
                'tier2_granular_achieved': False  # Manual edits may not maintain optimality
            }
            
            balance_metrics['leaning'] = leaning_balance
        
        # Gender balance
        if 'Gender' in edited_df.columns:
            overall_male = edited_df['Gender'].isin(['M', 'Male', 'male', 'MALE']).sum()
            overall_female = edited_df['Gender'].isin(['F', 'Female', 'female', 'FEMALE']).sum()
            
            gender_balance = {
                'overall_male': overall_male,
                'overall_female': overall_female,
                'male_percentage': round(overall_male / len(edited_df) * 100, 2),
                'tier2_achieved': False  # Manual edits may not maintain optimality
            }
            
            balance_metrics['gender'] = gender_balance
        
        updated_results['balance_metrics'] = balance_metrics
        
        # Update solution quality to indicate manual editing
        if 'solution_quality' in updated_results:
            updated_results['solution_quality']['status'] = "Manually Edited"
            updated_results['solution_quality']['notes'] = "Assignments were manually adjusted after hierarchical optimization"
            
            # Update hierarchical balance to reflect that manual edits may have changed optimality
            if 'hierarchical_balance' in updated_results['solution_quality']:
                updated_results['solution_quality']['hierarchical_balance']['manually_edited'] = True
        
        print(f"Successfully loaded edited assignments from {assignments_file_path}")
        print(f"Updated results reflect manual adjustments (optimality may no longer be guaranteed)")
        
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