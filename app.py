from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
import os
import pandas as pd
from werkzeug.utils import secure_filename
import json
import uuid
from datetime import datetime
import shutil
import numpy as np

# Import configuration
from config import UPLOAD_FOLDER, RESULTS_FOLDER, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH

# Import your modules
from modules.data_processing import process_juror_data
from modules.optimization import optimize_jury_assignment, format_results_for_output
from modules.utils import (
    visualize_jury_demographics,
    export_results_to_excel,
    create_html_report,
    analyze_optimization_results,
    export_assignments_for_editing,
    load_edited_assignments
)

app = Flask(__name__)
# In app.py, after creating the Flask app
def jury_letter(num):
    """Convert a numeric jury ID to a letter (1->A, 2->B, etc.)"""
    try:
        num = int(num)
        return chr(64 + num)
    except (ValueError, TypeError):
        return str(num)  # Return as-is if conversion fails

app.jinja_env.globals['jury_letter'] = jury_letter

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = 'your_secret_key_here'  # Replace with a real secret key in production

def json_numpy_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_session_folder():
    """Create a unique folder for this session's results"""
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    session_folder = os.path.join(app.config['RESULTS_FOLDER'], session_id)
    os.makedirs(session_folder, exist_ok=True)
    return session_folder

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # Check if user submitted an empty form
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    # Check if the file is allowed
    if file and allowed_file(file.filename):
        # Create a session folder for results
        session_folder = create_session_folder()
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(session_folder, filename)
        file.save(file_path)
        
        # Store file path in session
        session['file_path'] = file_path
        
        # Get form data
        num_juries = int(request.form.get('num_juries', 2))
        jury_size = int(request.form.get('jury_size', 12))
        
        # Get demographic variable rankings
        rankings = {
            'Final_Leaning': 5.0,  # Always highest priority - using underscore version
            'Gender': 4.5,
            'Race': float(5 - int(request.form.get('race_rank', 1))),
            'AgeGroup': float(5 - int(request.form.get('age_rank', 2))),
            'Education': float(5 - int(request.form.get('education_rank', 3))),
            'Marital': float(5 - int(request.form.get('marital_rank', 4)))
        }
        
        # Store parameters in session
        session['num_juries'] = num_juries
        session['jury_size'] = jury_size
        session['rankings'] = rankings
        
        # Process data and run optimization
        try:
            print("Starting data processing...")
            # Process the data
            data_dict = process_juror_data(file_path, num_juries, jury_size)
            
            print("Data processed successfully, now converting column names...")
            # Convert column names with spaces to underscore format for consistency in templates
            if 'original_data' in data_dict and isinstance(data_dict['original_data'], pd.DataFrame):
                print("Original column names:", data_dict['original_data'].columns.tolist())
                # Only replace spaces if the column name doesn't already have underscores
                data_dict['original_data'].columns = [col.replace(' ', '_') for col in data_dict['original_data'].columns]
                print("Updated column names:", data_dict['original_data'].columns.tolist())
            
            print("Running optimization with rankings:", rankings)
            # Run optimization
            results = optimize_jury_assignment(data_dict, rankings)
            print("Optimization completed successfully.")
            
            print("Saving results to JSON...")
            # Save results to a JSON file
            results_file = os.path.join(session_folder, 'results.json')
            
            # Convert DataFrame to dict for JSON serialization
            if 'summary' in results:
                results['summary'] = results['summary'].to_dict('records')
            if 'detailed_assignments' in results:
                results['detailed_assignments'] = results['detailed_assignments'].to_dict('records')
            
            # Custom function to handle numpy types when serializing to JSON
            def json_numpy_serializer(obj):
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            # Use the custom serializer
            with open(results_file, 'w') as f:
                json.dump(results, f, default=json_numpy_serializer)
            
            print("Exporting jury assignments for editing...")
            # Export jury assignments for editing
            edit_file_path = os.path.join(session_folder, "jury_assignments_for_editing.xlsx")
            export_assignments_for_editing(results, edit_file_path)
            session['edit_file_path'] = edit_file_path
            
            print("Redirecting to edit assignments page...")
            # Redirect to edit page
            return redirect(url_for('edit_assignments'))
        
        except Exception as e:
            import traceback
            print(f"ERROR: {str(e)}")
            print(traceback.format_exc())
            flash(f"Error processing data: {str(e)}")
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload an Excel file (.xlsx or .xls)')
    return redirect(url_for('index'))

@app.route('/edit')
def edit_assignments():
    # Check if we have a session
    if 'session_id' not in session:
        flash('Session expired. Please start again.')
        return redirect(url_for('index'))
    
    # Load the results from JSON
    session_folder = os.path.join(app.config['RESULTS_FOLDER'], session['session_id'])
    results_file = os.path.join(session_folder, 'results.json')
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Convert assignments back to DataFrame for the template
    assignments = pd.DataFrame(results['detailed_assignments'])
    
    # Get unique juries for the dropdown
    juries = sorted([j for j in assignments['jury'].unique() if j != 'Unassigned'])
    
    return render_template('edit.html', 
                          assignments=assignments.to_dict('records'), 
                          juries=juries)

@app.route('/save_edits', methods=['POST'])
def save_edits():
    # Add debugging output
    print("Save edits route called")
    
    # Check if we have a session
    if 'session_id' not in session:
        print("No session found")
        return {'status': 'error', 'message': 'Session expired'}, 400
    
    try:
        # Get the edited assignments from the form
        edited_data = request.get_json()
        if not edited_data:
            print("No data received")
            return {'status': 'error', 'message': 'No data received'}, 400
            
        print(f"Received edited data with {len(edited_data)} entries")
        
        # Convert to DataFrame
        edited_df = pd.DataFrame(edited_data)
        
        # Store the edited assignments directly in the session
        session['edited_assignments'] = edited_data
        
        # Get session folder path
        session_folder = os.path.join(app.config['RESULTS_FOLDER'], session['session_id'])
        
        # Save edited assignments to a JSON file
        edited_file_path = os.path.join(session_folder, "jury_assignments_edited.json")
        with open(edited_file_path, 'w') as f:
            json.dump(edited_data, f, default=json_numpy_serializer)
        
        # Load original results
        results_file = os.path.join(session_folder, 'results.json')
        with open(results_file, 'r') as f:
            original_results = json.load(f)
        
        # Replace the detailed assignments
        original_results['detailed_assignments'] = edited_data
        
        # Save updated results
        updated_results_file = os.path.join(session_folder, 'results_updated.json')
        with open(updated_results_file, 'w') as f:
            json.dump(original_results, f, default=json_numpy_serializer)
        
        session['updated_results_file'] = updated_results_file
        print(f"Saved updated results to {updated_results_file}")
        
        # Also save a temporary Excel file of the edited assignments
        temp_excel_path = os.path.join(session_folder, "jury_assignments_edited.xlsx")
        edited_df.to_excel(temp_excel_path, index=False)
        
        return {'status': 'success', 'message': 'Changes saved successfully'}
    except Exception as e:
        import traceback
        print(f"Error in save_edits: {str(e)}")
        print(traceback.format_exc())
        return {'status': 'error', 'message': str(e)}, 500

@app.route('/visualizations/<session_id>/figures/<filename>')
def serve_visualization(session_id, filename):
    """
    Serve visualization images from the session results folder
    """
    session_folder = os.path.join(app.config['RESULTS_FOLDER'], session_id)
    figures_path = os.path.join(session_folder, 'figures', filename)
    
    if os.path.exists(figures_path):
        return send_file(figures_path)
    else:
        return "Image not found", 404
    
@app.route('/generate_report', methods=['POST'])
def generate_report():
    # Check if we have a session
    if 'session_id' not in session:
        flash('Session expired. Please start again.')
        return redirect(url_for('index'))
    
    session_folder = os.path.join(app.config['RESULTS_FOLDER'], session['session_id'])
    
    # Load the most recent results
    if 'updated_results_file' in session:
        results_file = session['updated_results_file']
        print(f"Using updated results from: {results_file}")
    else:
        results_file = os.path.join(session_folder, 'results.json')
        print(f"Using original results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Convert dict back to DataFrame for processing
    if 'summary' in results:
        results['summary'] = pd.DataFrame(results['summary'])
    if 'detailed_assignments' in results:
        edited_df = pd.DataFrame(results['detailed_assignments'])
        results['detailed_assignments'] = edited_df
        
        # Regenerate jury analysis with the edited assignments
        jury_analysis = {}
        jury_indices = edited_df['jury'].unique()
        
        for j in jury_indices:
            # Skip unassigned jurors in the main analysis
            if j == 'Unassigned':
                continue
                
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
            
        # Also add analysis for unassigned jurors
        unassigned_j = edited_df[edited_df['jury'] == 'Unassigned']
        if len(unassigned_j) > 0:
            unassigned_analysis = {
                'size': len(unassigned_j),
                'leaning': unassigned_j['Final_Leaning'].value_counts().to_dict() if 'Final_Leaning' in unassigned_j.columns else {},
                'gender': unassigned_j['Gender'].value_counts().to_dict() if 'Gender' in unassigned_j.columns else {},
                'race': unassigned_j['Race'].value_counts().to_dict() if 'Race' in unassigned_j.columns else {},
                'age_group': unassigned_j['AgeGroup'].value_counts().to_dict() if 'AgeGroup' in unassigned_j.columns else {},
                'education': unassigned_j['Education'].value_counts().to_dict() if 'Education' in unassigned_j.columns else {},
                'marital': unassigned_j['Marital'].value_counts().to_dict() if 'Marital' in unassigned_j.columns else {},
                'jurors': unassigned_j.to_dict('records')
            }
            jury_analysis['Unassigned'] = unassigned_analysis
        
        # Update the jury analysis in the results
        results['jury_analysis'] = jury_analysis
        
        # Recalculate summary based on edited assignments
        from modules.optimization import format_results_for_output
        from modules.data_processing import process_juror_data
        
        # Dummy data_dict with original_data
        data_dict = {'original_data': edited_df}
        
        # Filter out unassigned jurors for the summary and export
        jury_assignments_fixed = edited_df[edited_df['jury'] != 'Unassigned'].copy()
        if len(jury_assignments_fixed) > 0:
            # If jury column contains letters (A, B, C...), convert to numbers (1, 2, 3...)
            jury_assignments_fixed['jury'] = jury_assignments_fixed['jury'].apply(
                lambda x: ord(x) - 64 if isinstance(x, str) and len(x) == 1 and x.isalpha() 
                else x
            )

            # Use the fixed assignments
            updated_results = format_results_for_output({
                'assignments': jury_assignments_fixed,
                'jury_analysis': {k: v for k, v in jury_analysis.items() if k != 'Unassigned'},
                'deviations': results['solution_quality']['deviations'],
                'solution_status': 'Manually Edited',
                'objective_value': results['solution_quality']['objective_value']
            }, data_dict)
            
            # Update the summary in results
            if 'summary' in updated_results:
                results['summary'] = updated_results['summary']
    
    # Create HTML report with embedded visualizations
    html_report_path = os.path.join(session_folder, 'jury_report.html')
    create_html_report(results, html_report_path, include_visualizations=True)
    
    # Store HTML report path in session
    session['html_report_path'] = html_report_path
    
    # Also export to Excel for detailed data
    excel_path = os.path.join(session_folder, 'jury_assignments.xlsx')
    export_results_to_excel(results, excel_path)
    session['excel_path'] = excel_path
    
    # Redirect to success page
    return redirect(url_for('success'))

@app.route('/success')
def success():
    # Check if we have a session 
    if 'session_id' not in session:
        flash('Session expired. Please start again.')
        return redirect(url_for('index'))
    
    # No need to check for pdf_report_path since we're using HTML
    return render_template('success.html')

@app.route('/download/<filetype>')
def download_file(filetype):
    # Check if we have a session
    if 'session_id' not in session:
        flash('Session expired. Please start again.')
        return redirect(url_for('index'))
    
    if filetype == 'html' and 'html_report_path' in session:
        return send_file(session['html_report_path'], as_attachment=True, 
                        download_name='jury_report.html', mimetype='text/html')
    
    elif filetype == 'excel' and 'excel_path' in session:
        return send_file(session['excel_path'], as_attachment=True, 
                        download_name='jury_assignments.xlsx', 
                        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    
    else:
        flash('File not found')
        return redirect(url_for('success'))

# Run the application
if __name__ == '__main__':
    app.run(debug=True)

app.debug = False