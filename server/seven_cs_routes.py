import json
import logging
from flask import Blueprint, request, jsonify
from app import db
from tables.seven_cs_analysis import SevenCsAnalysis
from tables.seven_cs_coded_segment import SevenCsCodedSegment
import database as db_helper
from seven_cs_service import analyze_session_seven_cs, update_seven_cs_analysis

seven_cs_bp = Blueprint('seven_cs', __name__)

@seven_cs_bp.route('/api/v1/seven-cs/analyze/<int:session_device_id>', methods=['POST'])
def trigger_analysis(session_device_id):
    """
    Manually trigger 7C analysis for a session device.
    This endpoint can be used to re-run analysis or update existing results.
    """
    try:
        # Check if session device exists
        session_device = db_helper.get_session_devices(id=session_device_id)
        if not session_device:
            return jsonify({'error': 'Session device not found'}), 404

        # Check if analysis is already in progress
        existing = db.session.query(SevenCsAnalysis).filter_by(
            session_device_id=session_device_id,
            analysis_status='processing'
        ).first()

        if existing:
            return jsonify({
                'status': 'processing',
                'message': 'Analysis already in progress',
                'analysis_id': existing.id
            }), 202

        # Trigger new analysis
        analysis = update_seven_cs_analysis(session_device_id)

        if analysis:
            return jsonify({
                'status': 'triggered',
                'message': 'Analysis started successfully',
                'analysis_id': analysis.id
            }), 202
        else:
            return jsonify({
                'error': 'Failed to start analysis'
            }), 500

    except Exception as e:
        logging.error(f"Error triggering 7C analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@seven_cs_bp.route('/api/v1/seven-cs/results/<int:session_device_id>', methods=['GET'])
def get_results(session_device_id):
    """
    Get the latest 7C analysis results for a session device.
    Returns overall scores, coded segments, and aggregated counts.
    """
    try:
        # Get the latest completed analysis
        analysis = db.session.query(SevenCsAnalysis).filter_by(
            session_device_id=session_device_id
        ).order_by(SevenCsAnalysis.created_at.desc()).first()

        if not analysis:
            return jsonify({
                'status': 'not_analyzed',
                'message': 'No analysis found for this session'
            }), 200

        # Get coded segments
        segments = db.session.query(SevenCsCodedSegment).filter_by(
            analysis_id=analysis.id
        ).all()

        # Calculate dimension counts
        dimension_counts = {}
        for segment in segments:
            if segment.dimension not in dimension_counts:
                dimension_counts[segment.dimension] = 0
            dimension_counts[segment.dimension] += 1

        # Prepare segments data
        segments_data = []
        for segment in segments:
            segments_data.append({
                'id': segment.id,
                'dimension': segment.dimension,
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'text_snippet': segment.text_snippet,
                'speaker_tag': segment.speaker_tag,
                'coding_reason': segment.coding_reason,
                'confidence': segment.confidence
            })

        # Return comprehensive results
        return jsonify({
            'status': analysis.analysis_status,
            'summary': analysis.analysis_summary,
            'counts': dimension_counts,
            'segments': segments_data,
            'metadata': {
                'total_segments_analyzed': analysis.total_segments_analyzed,
                'processing_time_seconds': analysis.processing_time_seconds,
                'tokens_used': analysis.tokens_used,
                'created_at': analysis.created_at.isoformat() if analysis.created_at else None,
                'updated_at': analysis.updated_at.isoformat() if analysis.updated_at else None
            }
        })

    except Exception as e:
        logging.error(f"Error getting 7C results: {str(e)}")
        return jsonify({'error': str(e)}), 500

@seven_cs_bp.route('/api/v1/seven-cs/status/<int:session_device_id>', methods=['GET'])
def check_status(session_device_id):
    """
    Check the status of 7C analysis for a session device.
    Useful for polling while analysis is in progress.
    """
    try:
        # Get the latest analysis
        analysis = db.session.query(SevenCsAnalysis).filter_by(
            session_device_id=session_device_id
        ).order_by(SevenCsAnalysis.created_at.desc()).first()

        if not analysis:
            return jsonify({
                'status': 'not_analyzed',
                'message': 'No analysis found'
            })

        return jsonify({
            'status': analysis.analysis_status,
            'analysis_id': analysis.id,
            'created_at': analysis.created_at.isoformat() if analysis.created_at else None,
            'updated_at': analysis.updated_at.isoformat() if analysis.updated_at else None
        })

    except Exception as e:
        logging.error(f"Error checking 7C status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@seven_cs_bp.route('/api/v1/seven-cs/segments/<int:session_device_id>/<dimension>', methods=['GET'])
def get_dimension_segments(session_device_id, dimension):
    """
    Get all coded segments for a specific dimension.
    Useful for displaying detailed evidence when a dimension is selected.
    """
    try:
        # Validate dimension
        valid_dimensions = ['climate', 'communication', 'compatibility', 'conflict',
                          'context', 'contribution', 'constructive']
        if dimension.lower() not in valid_dimensions:
            return jsonify({'error': 'Invalid dimension'}), 400

        # Get the latest analysis
        analysis = db.session.query(SevenCsAnalysis).filter_by(
            session_device_id=session_device_id,
            analysis_status='completed'
        ).order_by(SevenCsAnalysis.created_at.desc()).first()

        if not analysis:
            return jsonify({
                'status': 'not_analyzed',
                'message': 'No completed analysis found'
            }), 404

        # Get segments for this dimension
        segments = db.session.query(SevenCsCodedSegment).filter_by(
            analysis_id=analysis.id,
            dimension=dimension.lower()
        ).order_by(SevenCsCodedSegment.start_time).all()

        segments_data = []
        for segment in segments:
            segments_data.append({
                'id': segment.id,
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'text_snippet': segment.text_snippet,
                'speaker_tag': segment.speaker_tag,
                'coding_reason': segment.coding_reason,
                'confidence': segment.confidence,
                'transcript_id': segment.transcript_id
            })

        return jsonify({
            'dimension': dimension,
            'count': len(segments_data),
            'segments': segments_data
        })

    except Exception as e:
        logging.error(f"Error getting dimension segments: {str(e)}")
        return jsonify({'error': str(e)}), 500

@seven_cs_bp.route('/api/v1/seven-cs/export/<int:session_device_id>', methods=['GET'])
def export_analysis(session_device_id):
    """
    Export 7C analysis results in a format suitable for external analysis tools.
    """
    try:
        # Get the latest completed analysis
        analysis = db.session.query(SevenCsAnalysis).filter_by(
            session_device_id=session_device_id,
            analysis_status='completed'
        ).order_by(SevenCsAnalysis.created_at.desc()).first()

        if not analysis:
            return jsonify({
                'status': 'not_analyzed',
                'message': 'No completed analysis found'
            }), 404

        # Get all coded segments
        segments = db.session.query(SevenCsCodedSegment).filter_by(
            analysis_id=analysis.id
        ).order_by(SevenCsCodedSegment.start_time).all()

        # Get session info
        session_device = db_helper.get_session_devices(id=session_device_id)
        session = db_helper.get_sessions(id=session_device.session_id) if session_device else None

        # Prepare export data
        export_data = {
            'metadata': {
                'session_id': session.id if session else None,
                'session_name': session.name if session else None,
                'device_id': session_device_id,
                'device_name': session_device.name if session_device else None,
                'analysis_date': analysis.created_at.isoformat() if analysis.created_at else None,
                'total_segments_coded': len(segments)
            },
            'overall_scores': analysis.analysis_summary,
            'coded_segments': []
        }

        # Add segments
        for segment in segments:
            export_data['coded_segments'].append({
                'dimension': segment.dimension,
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'duration': segment.end_time - segment.start_time,
                'text': segment.text_snippet,
                'speaker': segment.speaker_tag,
                'coding_reason': segment.coding_reason,
                'confidence': segment.confidence
            })

        # Calculate summary statistics
        dimension_stats = {}
        for dim in ['climate', 'communication', 'compatibility', 'conflict',
                   'context', 'contribution', 'constructive']:
            dim_segments = [s for s in segments if s.dimension == dim]
            dimension_stats[dim] = {
                'count': len(dim_segments),
                'percentage': round((len(dim_segments) / len(segments) * 100), 2) if segments else 0,
                'avg_confidence': round(sum(s.confidence for s in dim_segments) / len(dim_segments), 2) if dim_segments else 0
            }

        export_data['dimension_statistics'] = dimension_stats

        return jsonify(export_data)

    except Exception as e:
        logging.error(f"Error exporting 7C analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500