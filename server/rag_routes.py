# server/rag_routes.py - Unified Implementation

from flask import Blueprint, request, jsonify
from rag_service import RAGService
from rag_query_parser import QueryParser
import logging

logger = logging.getLogger(__name__)

# Create Blueprint
rag_api = Blueprint('rag_api', __name__)

# Initialize services (shared instances)
rag_service = RAGService()
query_parser = QueryParser()


@rag_api.route('/api/v1/rag/search', methods=['POST'])
def unified_search():
    """
    Unified search endpoint - handles all query types intelligently
    
    Request body:
    {
        "query": "string",
        "filter_to_user": false,  // Optional: filter to user's sessions
        "n_results": 5  // Optional
    }
    
    Response varies by query type but always includes:
    {
        "query": "original query",
        "query_type": "retrieval|insights|comparison|similarity|temporal|error",
        "results": [...] or null,      // For retrieval queries
        "insights": "..." or null,      // Auto-generated for analytical queries
        "comparison": {...} or null,    // For comparative queries
        "similar": {...} or null,       // For similarity queries
        "timeline": {...} or null,      // For temporal queries
        "total_found": N,
        "error": "..." or null
    }
    """
    try:
        data = request.get_json()
        query = data.get('query', '')
        filter_to_user = data.get('filter_to_user', False)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Get user's session devices if filtering
        user_devices = None
        if filter_to_user:
            # TODO: Get actual user's sessions from auth
            user_devices = [448, 447, 446, 445]
        
        # Use QueryParser to handle the query intelligently
        result = query_parser.parse_and_execute(query, user_devices)
        
        # Normalize response format for frontend consistency
        response = {
            "query": query,
            "query_type": result.get('type', 'unknown'),
            "results": None,
            "insights": None,
            "comparison": None,
            "similar": None,
            "timeline": None,
            "total_found": 0,
            "error": None
        }
        
        query_type = result.get('type')
        
        if query_type == 'error':
            response['error'] = result.get('message', 'Unknown error')
            response['example'] = result.get('example')
            
        elif query_type in ['topic_search', 'pattern_analysis']:
            # Simple retrieval - return results
            search_results = result.get('results', {})
            response['results'] = search_results.get('results', [])
            response['total_found'] = search_results.get('total_found', 0)
            
        elif query_type == 'insights':
            # Analytical query - insights auto-generated
            search_results = result.get('evidence', {})
            response['results'] = search_results.get('results', [])
            response['total_found'] = search_results.get('total_found', 0)
            response['insights'] = result.get('insights')
            
        elif query_type == 'comparative':
            # Comparison between sessions
            response['comparison'] = result.get('comparisons', {})
            
        elif query_type == 'similar':
            # Similar sessions
            response['similar'] = result.get('results', {})
            
        elif query_type == 'temporal':
            # Timeline data
            response['timeline'] = result.get('timeline', [])
            response['summary'] = result.get('summary', {})
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Unified search error: {e}", exc_info=True)
        return jsonify({
            "query": query if 'query' in locals() else '',
            "query_type": "error",
            "results": None,
            "insights": None,
            "comparison": None,
            "similar": None,
            "timeline": None,
            "total_found": 0,
            "error": str(e)
        }), 500


@rag_api.route('/api/v1/rag/insights', methods=['POST'])
def generate_insights():
    """
    Manual insights generation endpoint
    
    Request body:
    {
        "query": "string",
        "search_results": {
            "query": "...",
            "results": [...],
            "total_found": N
        }
    }
    """
    try:
        data = request.get_json()
        
        query = data.get('query', '')
        search_results = data.get('search_results')
        
        if not query or not search_results:
            return jsonify({"error": "Query and search_results are required"}), 400
        
        insights = rag_service.generate_insights(query, search_results)
        
        return jsonify({
            "query": query,
            "insights": insights
        })
        
    except Exception as e:
        logger.error(f"Insights generation error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@rag_api.route('/api/v1/rag/stats', methods=['GET'])
def get_stats():
    """Get collection statistics"""
    try:
        stats = rag_service.get_collection_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": str(e)}), 500


@rag_api.route('/api/v1/rag/suggestions', methods=['GET'])
def get_suggestions():
    """Get suggested queries for UI"""
    try:
        return jsonify({
            "suggestions": query_parser.get_suggested_queries()
        })
    except Exception as e:
        logger.error(f"Suggestions error: {e}")
        return jsonify({"error": str(e)}), 500