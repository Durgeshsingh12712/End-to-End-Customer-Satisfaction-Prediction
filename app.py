import os, traceback
from pathlib import Path
from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    flash,
    redirect,
    url_for
)

from customerSatisfactionPrediction.loggers import logger
from customerSatisfactionPrediction.pipeline import PredictionPipeline

try:
    prediction_pipeline = PredictionPipeline()
    PIPELINE_LOADED = True
    logger.info(f"Prediction Pipeline Loaded Successfully")
except Exception as e:
    logger.info(f"Failed to Load prediction pipeline: {e}")
    prediction_pipeline = None
    PIPELINE_LOADED = False

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')

TICKET_TYPES = [
    'Technical Issue',
    'Billing Question',
    'Product Inquiry',
    'Account Support',
    'Bug Report',
    'Feature Request',
    'General Support',
    'Refund Request',
    'Installation Help',
    'Configuration Issue'
]

@app.route('/')
def index():
    """Main Page with predictio form"""
    return render_template('index.html', ticket_types = TICKET_TYPES, pipeline_loaded=PIPELINE_LOADED)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle Prediction Requests"""
    try:
        if not PIPELINE_LOADED:
            return jsonify({
                'error': 'Prediction Pipeline not Availabel',
                'success': False
            }), 500
        
        # Get Form Data
        ticket_type = request.form.get('ticket_type', '').strip()
        ticket_subject = request.form.get('ticket_subject', '').strip()
        ticket_description = request.form.get('ticket_description', '').strip()
        product_purchased = request.form.get('product_purchased', '').strip()

        # Validate Inputs
        if not all([ticket_type, ticket_subject, ticket_description]):
            flash('Please fill in all required fields (Ticket Type, Ticket Subject and Ticket Description)', 'error')
            return redirect(url_for('index'))
        
        logger.info(f"Making Prediction for Ticket: {ticket_type[:50]}...")

        result = prediction_pipeline.predict(
            ticket_type=ticket_type,
            ticket_subject=ticket_subject,
            ticket_description=ticket_description,
            product_purchased=product_purchased if product_purchased else None
        )

        # Add timestamp and input data to result
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result['input_data'] = {
            'ticket_type': ticket_type,
            'ticket_subject': ticket_subject,
            'ticket_description': ticket_description,
            'product_purchased': product_purchased
        }

        logger.info(f"Prediction Successfully: {result['predicted_priority']} "
                    f"(Confidence: {result['confidence']:.3f})")
        
        return render_template('result.html', result=result, ticket_type=TICKET_TYPES)
    
    except ValueError as ve:
        logger.error(f"Validation Error: {ve}")
        flash(f"Input Validation Error: {str(ve)}", 'error')
        return redirect(url_for('index'))
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        flash(f'An error occurred during prediction: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for Programmatic Predictions"""
    try:
        if not PIPELINE_LOADED:
            return jsonify({
                'error': 'Prediction Pipeline not Available',
                'success': False
            }), 500
        # Get JSON Data
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No JSON Data Provided',
                'success': False
            }), 400
        
        # Extract Require Fields
        ticket_type = data.get('ticket_type', '').strip()
        ticket_subject = data.get('ticket_subject', '').strip()
        ticket_description = data.get('ticket_description', '').strip()
        product_purchased = data.get('product_purchased', '').strip()

        # Validate input
        if not all([ticket_type, ticket_subject, ticket_description]):
            return jsonify({
                'error': 'Missing required Field: ticket_type, ticket_subject and ticket_description',
                'success': False
            }), 400
        
        # Make Prediction
        result = prediction_pipeline.predict(
            ticket_type=ticket_type,
            ticket_subject=ticket_subject,
            ticket_description=ticket_description,
            product_purchased=product_purchased if product_purchased else None
        )

        result['success'] = True
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(result)

    except ValueError as ve:
        return jsonify({
            'error': f'Validation error: {str(ve)}',
            'success': False
        }), 400
    
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 500


@app.route('/model-info')
def model_info():
    """Display model information"""
    try:
        if not PIPELINE_LOADED:
            info = {
                'status': 'Pipeline not loaded',
                'error': 'Prediction pipeline could not be initialized'
            }
        else:
            info = prediction_pipeline.get_model_info()
            info['status'] = 'Loaded successfully'
        
        return render_template('model_info.html', model_info=info)
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return render_template('model_info.html', 
                             model_info={'status': 'Error', 'error': str(e)})
    
@app.route('/health')
def health_check():
    """Health Check Endpoint"""
    return jsonify({
        'status': 'healthy' if PIPELINE_LOADED else 'degraded',
        'pipeline_loaded': PIPELINE_LOADED,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)