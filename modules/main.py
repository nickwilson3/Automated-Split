# main.py
import argparse
import os
import json
from datetime import datetime
import pandas as pd

# Import project modules
from data_processing import process_juror_data
from optimization import optimize_jury_assignment
from utils import (
    visualize_jury_demographics,
    export_results_to_excel,
    create_html_report,
    analyze_optimization_results,
    export_assignments_for_editing,
    load_edited_assignments,
    manual_adjustment_workflow
)


def create_output_directory(output_base="results"):
    """
    Create a timestamped output directory for results.
    
    Parameters:
    output_base (str): Base directory name
    
    Returns:
    str: Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base, f"jury_optimization_{timestamp}")
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir


def load_config(config_path):
    """
    Load configuration from a JSON file.
    
    Parameters:
    config_path (str): Path to the configuration file
    
    Returns:
    dict: Configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"Successfully loaded configuration from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        print("Using default configuration...")
        return {}


def save_config(config, output_dir):
    """
    Save the current configuration to a JSON file.
    
    Parameters:
    config (dict): Configuration to save
    output_dir (str): Directory to save the configuration
    
    Returns:
    str: Path to the saved configuration file
    """
    config_path = os.path.join(output_dir, "config.json")
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Configuration saved to {config_path}")
        return config_path
    except Exception as e:
        print(f"Error saving configuration: {str(e)}")
        return None


def run_optimization_pipeline(
    file_path, 
    num_juries, 
    jury_size, 
    demographic_vars=None,
    priority_weights=None,
    time_limit=300,
    output_dir=None,
    export_excel=True,
    create_report=True,
    create_visualizations=True,
    allow_manual_edit=False
):
    """
    Run the complete jury selection optimization pipeline.
    
    Parameters:
    file_path (str): Path to the input data file
    num_juries (int): Number of juries to form
    jury_size (int): Size of each jury
    demographic_vars (list, optional): List of demographic variables to consider
    priority_weights (dict, optional): Priority weights for demographic variables
    time_limit (int): Time limit for solver in seconds
    output_dir (str, optional): Directory to save results
    export_excel (bool): Whether to export results to Excel
    create_report (bool): Whether to create an HTML report
    create_visualizations (bool): Whether to create visualization figures
    
    Returns:
    dict: Complete results and output paths
    """
    # Create output directory if not provided
    if output_dir is None:
        output_dir = create_output_directory()
    
    # Set default demographic variables if not provided
    if demographic_vars is None:
        demographic_vars = ['Final Leaning', 'Race', 'Age', 'Education', 'Marital']
    
    # Set default priority weights if not provided
    if priority_weights is None:
        priority_weights = {
            'Final Leaning': 5.0,
            'Race': 4.0,
            'AgeGroup': 3.0,
            'Education': 2.0,
            'Marital': 1.0
        }
    
    # Save configuration
    config = {
        'file_path': file_path,
        'num_juries': num_juries,
        'jury_size': jury_size,
        'demographic_vars': demographic_vars,
        'priority_weights': priority_weights,
        'time_limit': time_limit
    }
    save_config(config, output_dir)
    
    # Process the data
    print(f"Processing data from {file_path}...")
    try:
        data_dict = process_juror_data(file_path, num_juries, jury_size, demographic_vars)
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return {'error': str(e), 'stage': 'data_processing'}
    
    # Print some data summary
    print(f"Processed {data_dict['summary']['total_jurors']} jurors.")
    print(f"Feasibility check: {data_dict['feasibility_message']}")
    
    # Run optimization
    print("Running jury assignment optimization...")
    try:
        results = optimize_jury_assignment(data_dict, priority_weights, time_limit)
    except Exception as e:
        print(f"Error in optimization: {str(e)}")
        return {'error': str(e), 'stage': 'optimization', 'data_dict': data_dict}
    
    if allow_manual_edit:
        print("\n--- Manual Adjustment Phase ---")
        edit_file_path = os.path.join(output_dir, "jury_assignments_for_editing.xlsx")
        results = manual_adjustment_workflow(results, edit_file_path)
        print("Using manually adjusted assignments for reporting.\n")
    
    # Print optimization results summary
    print(f"Optimization completed with status: {results['solution_quality']['status']}")
    print(f"Objective value: {results['solution_quality']['objective_value']}")
    
    # Create output files
    output_files = {}
    
    # Export to Excel if requested
    if export_excel:
        print("Exporting results to Excel...")
        excel_path = os.path.join(output_dir, "jury_assignments.xlsx")
        try:
            excel_file = export_results_to_excel(results, excel_path)
            output_files['excel'] = excel_file
        except Exception as e:
            print(f"Error exporting to Excel: {str(e)}")
    
    # Create visualizations if requested
    viz_dir = None
    if create_visualizations:
        print("Creating visualizations...")
        viz_dir = os.path.join(output_dir, "visualizations")
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        try:
            figures = visualize_jury_demographics(results, viz_dir)
            output_files['visualizations'] = viz_dir
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
    
    # Create HTML report if requested
    if create_report:
        print("Creating HTML report...")
        report_path = os.path.join(output_dir, "jury_report.html")
        try:
            html_file = create_html_report(results, report_path, create_visualizations)
            output_files['html_report'] = html_file
        except Exception as e:
            print(f"Error creating HTML report: {str(e)}")
    
    # Analyze optimization results
    print("Analyzing optimization results...")
    try:
        analysis = analyze_optimization_results(results, data_dict)
        
        # Save analysis to JSON file
        analysis_path = os.path.join(output_dir, "analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=4)
        
        output_files['analysis'] = analysis_path
        
        # Print insights
        print("\nAnalysis insights:")
        for insight in analysis['insights']:
            print(f"- {insight}")
    except Exception as e:
        print(f"Error analyzing results: {str(e)}")
    
    # Return results and output paths
    return {
        'results': results,
        'data_dict': data_dict,
        'output_dir': output_dir,
        'output_files': output_files
    }


def parse_arguments():
    """
    Parse command line arguments for the main script.
    
    Returns:
    argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Jury Selection Optimization System')
    
    parser.add_argument('--file', '-f', required=True,
                        help='Path to the Excel file containing juror data')
    
    parser.add_argument('--num-juries', '-n', type=int, default=2,
                        help='Number of juries to form (default: 2)')
    
    parser.add_argument('--jury-size', '-s', type=int, default=12,
                        help='Size of each jury (default: 12)')
    
    parser.add_argument('--time-limit', '-t', type=int, default=300,
                        help='Time limit for solver in seconds (default: 300)')
    
    parser.add_argument('--config', '-c',
                        help='Path to configuration JSON file')
    
    parser.add_argument('--output-dir', '-o',
                        help='Directory to save results')
    
    parser.add_argument('--no-excel', action='store_true',
                        help='Do not export results to Excel')
    
    parser.add_argument('--no-report', action='store_true',
                        help='Do not create HTML report')
    
    parser.add_argument('--no-visualizations', action='store_true',
                        help='Do not create visualizations')
    parser.add_argument('--manual-edit', action='store_true',
                        help='Allow manual editing of jury assignments before generating reports')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration from file if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override configuration with command line arguments
    file_path = args.file
    num_juries = config.get('num_juries', args.num_juries)
    jury_size = config.get('jury_size', args.jury_size)
    time_limit = config.get('time_limit', args.time_limit)
    demographic_vars = config.get('demographic_vars', None)
    priority_weights = config.get('priority_weights', None)
    output_dir = args.output_dir if args.output_dir else None
    
    # Run the optimization pipeline
    results = run_optimization_pipeline(
        file_path=file_path,
        num_juries=num_juries,
        jury_size=jury_size,
        demographic_vars=demographic_vars,
        priority_weights=priority_weights,
        time_limit=time_limit,
        output_dir=output_dir,
        export_excel=not args.no_excel,
        create_report=not args.no_report,
        create_visualizations=not args.no_visualizations,
        allow_manual_edit=args.manual_edit
    )
    
    # Print output information
    if 'output_dir' in results:
        print(f"\nAll results saved to: {results['output_dir']}")
    
    if 'output_files' in results:
        print("\nOutput files:")
        for file_type, file_path in results['output_files'].items():
            print(f"- {file_type}: {file_path}")
    
    print("\nOptimization process completed.")