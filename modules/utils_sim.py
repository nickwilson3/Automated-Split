import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import seaborn as sns
import os
from datetime import datetime
import numpy as np


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
        output_path = f"jury_assignments_simultaneous_{timestamp}.xlsx"
    
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
                'Metric': ['Status', 'Optimization Method', 'Objective Value'],
                'Value': [
                    solution_quality['status'],
                    solution_quality.get('optimization_method', 'Simultaneous'),
                    solution_quality.get('objective_value', 'N/A')
                ]
            }
            
            # Add tier weights
            if 'tier_weights' in solution_quality:
                for tier, weight in solution_quality['tier_weights'].items():
                    sq_data['Metric'].append(f"Weight - {tier}")
                    sq_data['Value'].append(weight)
            
            # Add deviations
            if 'deviations' in solution_quality:
                for var, dev in solution_quality['deviations'].items():
                    sq_data['Metric'].append(f"Deviation - {var}")
                    sq_data['Value'].append(dev)
            
            sq_df = pd.DataFrame(sq_data)
            sq_df.to_excel(writer, sheet_name='Solution_Quality', index=False)
        
        # Export similarity metrics to a sheet
        if 'similarity_metrics' in results:
            similarity_metrics = results['similarity_metrics']
            
            # Pairwise differences
            if 'pairwise_differences' in similarity_metrics:
                pw_data = {
                    'Comparison': list(similarity_metrics['pairwise_differences'].keys()),
                    'Difference': list(similarity_metrics['pairwise_differences'].values())
                }
                pw_df = pd.DataFrame(pw_data)
                pw_df.to_excel(writer, sheet_name='Pairwise_Differences', index=False)
            
            # Max differences
            if 'max_difference' in similarity_metrics:
                max_data = {
                    'Demographic_Category': list(similarity_metrics['max_difference'].keys()),
                    'Max_Difference': list(similarity_metrics['max_difference'].values())
                }
                max_df = pd.DataFrame(max_data)
                max_df.to_excel(writer, sheet_name='Max_Differences', index=False)
        
        # Export individual jury analysis to separate sheets
        if 'jury_analysis' in results:
            jury_analysis = results['jury_analysis']
            
            for jury_num, analysis in jury_analysis.items():
                # Handle both numeric and alphabetic jury IDs
                if isinstance(jury_num, str) and jury_num.isalpha():
                    jury_label = jury_num
                elif jury_num == 'Unassigned':
                    jury_label = 'Unassigned'
                else:
                    try:
                        jury_label = chr(64 + int(jury_num))
                    except (ValueError, TypeError):
                        jury_label = str(jury_num)
                
                # Create a DataFrame for this jury's jurors
                if 'jurors' in analysis:
                    jury_df = pd.DataFrame(analysis['jurors'])
                    sheet_name = f'Jury_{jury_label}_Details'
                    jury_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
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
        output_path = f"jury_report_simultaneous_{timestamp}.html"
    
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
            if jury_num == 'Unassigned':
                continue
            elif isinstance(jury_num, str) and jury_num.isalpha():
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
            fig, ax = plt.subplots(figsize=(10, 6))
            leaning_df = pd.DataFrame(leaning_data)
            leaning_order = ['P+', 'P', 'D', 'D+']
            sns.barplot(x='Jury', y='Count', hue='Leaning', hue_order=leaning_order, data=leaning_df, ax=ax)
            ax.set_title('Juror Leaning Distribution by Jury (Simultaneous Optimization)')
            ax.set_ylabel('Number of Jurors')
            ax.legend(title='Leaning')
            
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            embedded_images['leaning'] = img_str
            
            plt.close(fig)
        
        # 2. Gender distribution
        gender_data = []
        for jury_num, analysis in jury_analysis.items():
            if jury_num == 'Unassigned':
                continue
            elif isinstance(jury_num, str) and jury_num.isalpha():
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
            if jury_num == 'Unassigned':
                continue
            elif isinstance(jury_num, str) and jury_num.isalpha():
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
            if jury_num == 'Unassigned':
                continue
            elif isinstance(jury_num, str) and jury_num.isalpha():
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
            if jury_num == 'Unassigned':
                continue
            elif isinstance(jury_num, str) and jury_num.isalpha():
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
        
        # 6. Similarity heatmap (pairwise differences)
        if 'similarity_metrics' in results and 'pairwise_differences' in results['similarity_metrics']:
            pw_diff = results['similarity_metrics']['pairwise_differences']
            
            # Create a matrix of pairwise differences
            # Extract jury pairs and values
            if pw_diff:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Convert pairwise differences to a format suitable for display
                diff_data = []
                for key, val in pw_diff.items():
                    parts = key.split('_')
                    if len(parts) >= 3:
                        demo = parts[0]
                        pair = '_'.join(parts[1:])
                        diff_data.append({
                            'Demographic': demo,
                            'Jury_Pair': pair,
                            'Difference': val
                        })
                
                if diff_data:
                    diff_df = pd.DataFrame(diff_data)
                    pivot_df = diff_df.pivot(index='Demographic', columns='Jury_Pair', values='Difference')
                    pivot_df = pivot_df.fillna(0)
                    
                    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
                    ax.set_title('Pairwise Differences Between Juries (Lower is Better)')
                    
                    buffer = BytesIO()
                    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)
                    
                    img_str = base64.b64encode(buffer.read()).decode('utf-8')
                    embedded_images['similarity'] = img_str
                    
                    plt.close(fig)
    
    # Generate HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jury Selection Optimization Report - Simultaneous</title>
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
            .optimization-info { background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 15px 0; }
            .tier-info { background-color: #f3e5f5; padding: 10px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>Jury Selection Optimization Report</h1>
        <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        
        <div class="section">
            <h2>Simultaneous Optimization Summary</h2>
            <div class="optimization-info">
                <h4>Optimization Approach:</h4>
                <p><strong>Method:</strong> Simultaneous optimization of all juries</p>
                <p><strong>Goal:</strong> Minimize differences between juries across all demographics to create maximally similar jury compositions</p>
                <p><strong>Advantage:</strong> Ensures all juries are as similar as possible rather than prioritizing one jury over others</p>
            </div>
            <div class="tier-info">
                <h4>Hierarchical Weighting System:</h4>
                <p><strong>TIER 1 (Weight ~1000):</strong> P/D overall balance similarity - ensures juries have similar numbers of P-leaning and D-leaning jurors</p>
                <p><strong>TIER 2 (Weight ~90-100):</strong> Granular P+/P/D/D+ similarity and gender balance - fine-tunes leaning distribution and gender balance</p>
                <p><strong>TIER 3 (Weight 1-10):</strong> Other demographics - balances race, age, education, and marital status based on user preferences</p>
            </div>
    """
    
    # Add solution quality information
    if 'solution_quality' in results:
        solution_quality = results['solution_quality']
        html_content += f"""
            <p><span class="metric">Solution Status:</span> {solution_quality['status']}</p>
            <p><span class="metric">Objective Value:</span> {solution_quality.get('objective_value', 'N/A'):.2f} (lower is better)</p>
            <p><span class="metric">Optimization Method:</span> {solution_quality.get('optimization_method', 'Simultaneous')}</p>
        """
        
        # Add tier weights used
        if 'tier_weights' in solution_quality:
            html_content += "<h3>Tier Weights Applied</h3>"
            html_content += "<table>"
            html_content += "<tr><th>Tier/Demographic</th><th>Weight</th></tr>"
            for tier, weight in solution_quality['tier_weights'].items():
                html_content += f"<tr><td>{tier.replace('_', ' ').title()}</td><td>{weight}</td></tr>"
            html_content += "</table>"
        
        # Add deviations if available
        if 'deviations' in solution_quality and solution_quality['deviations']:
            html_content += "<h3>Deviations Summary (Total Weighted Penalties)</h3>"
            html_content += "<table>"
            html_content += "<tr><th>Demographic Comparison</th><th>Total Deviation</th></tr>"
            
            # Sort by deviation value (highest first)
            sorted_devs = sorted(solution_quality['deviations'].items(), key=lambda x: x[1], reverse=True)
            
            for var, dev in sorted_devs[:20]:  # Show top 20
                html_content += f"<tr><td>{var}</td><td>{dev:.2f}</td></tr>"
            
            html_content += "</table>"
    
    # Add similarity metrics
    if 'similarity_metrics' in results:
        similarity_metrics = results['similarity_metrics']
        
        if 'pairwise_differences' in similarity_metrics and similarity_metrics['pairwise_differences']:
            html_content += "<h3>Pairwise Jury Differences (Absolute Counts)</h3>"
            html_content += "<table>"
            html_content += "<tr><th>Comparison</th><th>Difference</th></tr>"
            
            for comp, diff in similarity_metrics['pairwise_differences'].items():
                html_content += f"<tr><td>{comp.replace('_', ' ')}</td><td>{diff}</td></tr>"
            
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
            'similarity': 'Pairwise Similarity Heatmap'
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
            if jury_num == 'Unassigned':
                jury_label = 'Unassigned'
            elif isinstance(jury_num, str) and jury_num.isalpha():
                jury_label = jury_num
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
                
                # Select relevant columns
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


def analyze_optimization_results(results):
    """
    Analyze the simultaneous optimization results to provide insights.
    
    Parameters:
    results (dict): Formatted results from the optimization process
    
    Returns:
    dict: Analysis metrics and insights
    """
    analysis = {
        'metrics': {},
        'insights': [],
        'similarity_assessment': {}
    }
    
    # Analyze solution quality
    if 'solution_quality' in results:
        solution_quality = results['solution_quality']
        
        analysis['metrics']['objective_value'] = solution_quality.get('objective_value', None)
        analysis['metrics']['optimization_status'] = solution_quality.get('status', 'Unknown')
        
        if solution_quality.get('status') == 'Optimal':
            analysis['insights'].append("Optimal solution found: All juries optimized simultaneously for maximum similarity")
        elif solution_quality.get('status') == 'Feasible':
            analysis['insights'].append("Feasible solution found within time limit: Juries are well-balanced but may not be perfectly optimal")
    
    # Analyze similarity metrics
    if 'similarity_metrics' in results:
        similarity_metrics = results['similarity_metrics']
        
        # Pairwise differences
        if 'pairwise_differences' in similarity_metrics:
            pw_diffs = similarity_metrics['pairwise_differences']
            
            # Calculate average difference for P and D
            p_diffs = [v for k, v in pw_diffs.items() if k.startswith('P_')]
            d_diffs = [v for k, v in pw_diffs.items() if k.startswith('D_')]
            
            if p_diffs:
                avg_p_diff = sum(p_diffs) / len(p_diffs)
                max_p_diff = max(p_diffs)
                analysis['metrics']['avg_p_difference'] = avg_p_diff
                analysis['metrics']['max_p_difference'] = max_p_diff
                
                if max_p_diff == 0:
                    analysis['insights'].append("Perfect P-leaning balance achieved: All juries have identical P-leaning counts")
                elif max_p_diff <= 1:
                    analysis['insights'].append(f"Excellent P-leaning balance: Maximum difference of {max_p_diff} juror between juries")
                else:
                    analysis['insights'].append(f"P-leaning balance: Maximum difference of {max_p_diff} jurors between juries")
            
            if d_diffs:
                avg_d_diff = sum(d_diffs) / len(d_diffs)
                max_d_diff = max(d_diffs)
                analysis['metrics']['avg_d_difference'] = avg_d_diff
                analysis['metrics']['max_d_difference'] = max_d_diff
                
                if max_d_diff == 0:
                    analysis['insights'].append("Perfect D-leaning balance achieved: All juries have identical D-leaning counts")
                elif max_d_diff <= 1:
                    analysis['insights'].append(f"Excellent D-leaning balance: Maximum difference of {max_d_diff} juror between juries")
                else:
                    analysis['insights'].append(f"D-leaning balance: Maximum difference of {max_d_diff} jurors between juries")
        
        # Max differences across all demographics
        if 'max_difference' in similarity_metrics:
            max_diffs = similarity_metrics['max_difference']
            
            # Find demographics with perfect balance (max_diff = 0)
            perfect_demos = [k for k, v in max_diffs.items() if v == 0]
            if perfect_demos:
                analysis['insights'].append(f"Perfect balance achieved in {len(perfect_demos)} demographic categories")
            
            # Find demographics with largest imbalances
            worst_demos = sorted(max_diffs.items(), key=lambda x: x[1], reverse=True)[:3]
            if worst_demos and worst_demos[0][1] > 0:
                analysis['insights'].append(f"Largest imbalances in: {', '.join([d[0] for d in worst_demos])}")
    
    # Overall assessment
    if 'metrics' in analysis:
        if analysis['metrics'].get('max_p_difference', float('inf')) <= 1 and \
           analysis['metrics'].get('max_d_difference', float('inf')) <= 1:
            analysis['similarity_assessment']['overall_quality'] = 'Excellent'
            analysis['insights'].append("Overall assessment: Juries are highly similar with minimal differences")
        elif analysis['metrics'].get('max_p_difference', float('inf')) <= 2 and \
             analysis['metrics'].get('max_d_difference', float('inf')) <= 2:
            analysis['similarity_assessment']['overall_quality'] = 'Good'
            analysis['insights'].append("Overall assessment: Juries are well-balanced with acceptable differences")
        else:
            analysis['similarity_assessment']['overall_quality'] = 'Fair'
            analysis['insights'].append("Overall assessment: Juries are reasonably balanced but show some variation")
    
    return analysis


if __name__ == "__main__":
    # Example usage
    print("Utils module loaded successfully")
    print("Available functions:")
    print("  - export_results_to_excel()")
    print("  - create_html_report()")
    print("  - analyze_optimization_results()")