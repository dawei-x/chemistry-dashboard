from flask import Blueprint, request, jsonify
import logging
from datetime import datetime
from app import db
from tables.concept_session import ConceptSession
from tables.concept_node import ConceptNode
from tables.concept_edge import ConceptEdge
from tables.concept_cluster import ConceptCluster
import database as db_helper

# Create Blueprint for concept mapping routes
concept_bp = Blueprint('concepts', __name__)


@concept_bp.route('/api/v1/concepts/regenerate/<int:session_device_id>', methods=['POST'])
def regenerate_concepts(session_device_id):
    """
    Manually trigger concept map regeneration for a session device.
    This endpoint can be used to re-run extraction or update existing results.
    """
    try:
        # Check if session device exists
        session_device = db_helper.get_session_devices(id=session_device_id)
        if not session_device:
            return jsonify({'error': 'Session device not found'}), 404

        # Check if concept session exists and is already processing
        concept_session = ConceptSession.query.filter_by(
            session_device_id=session_device_id
        ).first()

        if concept_session and concept_session.generation_status == 'processing':
            return jsonify({
                'status': 'processing',
                'message': 'Concept generation already in progress',
                'concept_session_id': concept_session.id
            }), 202

        # Trigger regeneration
        from concept_generation_service import generate_concepts_for_session_device
        result = generate_concepts_for_session_device(session_device_id)

        if result:
            return jsonify({
                'status': 'completed',
                'message': 'Concepts regenerated successfully',
                'concept_session_id': result.id,
                'node_count': ConceptNode.query.filter_by(concept_session_id=result.id).count(),
                'edge_count': ConceptEdge.query.filter_by(concept_session_id=result.id).count()
            }), 200
        else:
            return jsonify({
                'error': 'Failed to regenerate concepts'
            }), 500

    except Exception as e:
        logging.error(f"Error regenerating concepts: {str(e)}")
        return jsonify({'error': str(e)}), 500


def _index_session_for_rag(session_device_id: int) -> bool:
    """
    Index a session for session-level RAG search after concept map updates.

    This is called automatically after clustering completes to update the
    session_summaries collection in ChromaDB with the latest concept map data.

    Returns True if indexing succeeded, False otherwise.
    """
    try:
        from session_serializer import SessionSerializer
        from rag_service import RAGService

        serializer = SessionSerializer()
        rag_service = RAGService()

        # Serialize the session data
        serialized = serializer.serialize_for_embedding(session_device_id)

        if not serialized:
            logging.warning(f"No data to index for session {session_device_id}")
            return False

        # Index in the session collection
        success = rag_service.index_session(session_device_id, serialized)

        if success:
            logging.info(f"Session {session_device_id} indexed for RAG - "
                        f"nodes: {serialized['metadata'].get('node_count', 0)}, "
                        f"transcripts: {serialized['metadata'].get('transcript_count', 0)}")

            # Re-index affected speakers for cross-session search
            _reindex_affected_speakers(session_device_id)
        else:
            logging.error(f"Failed to index session {session_device_id} for RAG")

        return success

    except Exception as e:
        logging.error(f"Error indexing session {session_device_id} for RAG: {e}", exc_info=True)
        return False


def _reindex_affected_speakers(session_device_id: int):
    """
    Re-index all speakers in a session after session data changes.

    Called after session indexing to keep speaker profiles up to date
    across all their sessions.
    """
    try:
        from tables.speaker import Speaker
        from speaker_serializer import SpeakerSerializer
        from rag_service import RAGService

        # Get all unique speaker aliases in this session
        speakers = Speaker.query.filter_by(session_device_id=session_device_id).all()
        aliases = set(s.alias for s in speakers if s.alias)

        if not aliases:
            return

        logging.info(f"Re-indexing {len(aliases)} speakers affected by session {session_device_id}")

        serializer = SpeakerSerializer()
        rag_service = RAGService()

        for alias in aliases:
            try:
                serialized = serializer.serialize_speaker(alias)
                if serialized:
                    rag_service.index_speaker(alias, serialized)
                    logging.debug(f"  Re-indexed speaker: {alias}")
            except Exception as e:
                logging.error(f"  Failed to re-index speaker {alias}: {e}")

    except Exception as e:
        logging.error(f"Error re-indexing speakers for session {session_device_id}: {e}", exc_info=True)

def parse_session_device_id(source):
    """Parse session_device_id from various formats
    
    Expected format: "sessionId:deviceId" (e.g., "123:456")
    Returns the session_device_id from the database
    """
    try:
        source_str = str(source)
        
        # Primary format: sessionId:deviceId
        if ':' in source_str:
            parts = source_str.split(':')
            if len(parts) == 2:
                session_id = int(parts[0])
                device_id = int(parts[1])
                
                # Query the database to find the actual session_device_id
                from tables.session_device import SessionDevice
                session_device = SessionDevice.query.filter_by(
                    session_id=session_id,
                    device_id=device_id
                ).first()
                
                if session_device:
                    return session_device.id
                else:
                    # Try to create a minimal SessionDevice for testing/development
                    try:
                        session_device = SessionDevice(
                            session_id=session_id,
                            device_id=device_id,
                            name=f"Device_{device_id}"
                        )
                        db.session.add(session_device)
                        db.session.commit()
                        logging.info(f"Created new session_device with id={session_device.id}")
                        return session_device.id
                    except Exception as e:
                        logging.error(f"Failed to create session_device: {e}")
                        db.session.rollback()
                        return None
        
        # Fallback: assume it's already a session_device_id
        return int(source_str)
        
    except (ValueError, IndexError) as e:
        logging.error(f"Failed to parse source '{source}': {e}")
        return None

@concept_bp.route('/api/v1/concepts', methods=['POST'])
def receive_concept_update():
    """Receive concept updates from audio processing"""
    try:
        data = request.get_json()
        source = data.get('source')
        concept_update = data.get('concept_update')
        timestamp = data.get('timestamp')
        logging.info(f"Received timestamp: {timestamp}")
        
        # Validate input
        if not all([source, concept_update, timestamp]):
            logging.error(f"Missing required fields. source={source is not None}, "
                         f"concept_update={concept_update is not None}, timestamp={timestamp is not None}")
            return jsonify({'error': 'Missing required fields: source, concept_update, or timestamp'}), 400
        
        # Parse to get session_device_id
        session_device_id = parse_session_device_id(source)
        
        if session_device_id is None:
            logging.error(f"Could not find or create session_device for source: {source}")
            return jsonify({'error': 'Session device not found or could not be created'}), 404
            
        logging.info(f"Using session_device_id={session_device_id} for source={source}")
            
        # Find or create concept session
        concept_session = ConceptSession.query.filter_by(
            session_device_id=session_device_id
        ).first()
        
        if not concept_session:
            logging.info(f"Creating new concept session for session_device_id={session_device_id}")
            concept_session = ConceptSession(
                session_device_id=session_device_id,
                discourse_type=concept_update.get('discourse_type', 'exploratory')
            )
            db.session.add(concept_session)
            db.session.commit()
            logging.info(f"Created concept session with id={concept_session.id}")
        
        nodes_added = 0
        edges_added = 0
        
        # Add new nodes from the update
        if 'nodes' in concept_update and concept_update['nodes']:
            for node_data in concept_update['nodes']:
                node_id = node_data.get('id')
                if not node_id:
                    logging.warning(f"Skipping node without ID: {node_data}")
                    continue
                    
                # Check if node already exists
                existing = ConceptNode.query.filter_by(
                    id=node_id,
                    concept_session_id=concept_session.id
                ).first()
                
                if not existing:
                    # Check for text-based duplicates as safety net
                    node_text = node_data.get('text', '').strip()
                    if node_text:
                        text_duplicate = ConceptNode.query.filter_by(
                            concept_session_id=concept_session.id,
                            text=node_text
                        ).first()

                        if text_duplicate:
                            logging.info(f"Skipping duplicate concept text: '{node_text[:50]}...' (existing ID: {text_duplicate.id}, attempted ID: {node_id})")

                            # Update any edges that reference this duplicate to point to the existing node
                            for edge_data in edges:
                                if edge_data.get('source') == node_id:
                                    edge_data['source'] = text_duplicate.id
                                if edge_data.get('target') == node_id:
                                    edge_data['target'] = text_duplicate.id
                            continue

                    # Map 'type' to 'node_type' to match the schema
                    speaker_value = node_data.get('speaker') or node_data.get('speaker_id')
                    if speaker_value == 'Unknown' or speaker_value == '' or not speaker_value:
                        speaker_id = None  # NULL in database
                    elif isinstance(speaker_value, str) and speaker_value.isdigit():
                        speaker_id = int(speaker_value)
                    elif isinstance(speaker_value, int):
                        speaker_id = speaker_value
                    else:
                        speaker_id = None
                    node = ConceptNode(
                        id=node_id,
                        concept_session_id=concept_session.id,
                        text=node_data.get('text', ''),
                        node_type=node_data.get('type', 'concept'),  # 'type' from extractor -> 'node_type' in DB
                        speaker_id=speaker_id,
                        timestamp=timestamp or node_data.get('timestamp')
                    )
                    db.session.add(node)
                    nodes_added += 1
                    logging.debug(f"Added node: {node_id} - {node_data.get('text', '')[:50]}")
        
        # Add new edges from the update
        if 'edges' in concept_update and concept_update['edges']:
            for edge_data in concept_update['edges']:
                edge_id = edge_data.get('id')
                if not edge_id:
                    logging.warning(f"Skipping edge without ID: {edge_data}")
                    continue
                    
                existing = ConceptEdge.query.filter_by(
                    id=edge_id,
                    concept_session_id=concept_session.id
                ).first()
                
                if not existing:
                    source_id = edge_data.get('source')
                    target_id = edge_data.get('target')
                    
                    # Validate that source and target nodes exist
                    if not source_id or not target_id:
                        logging.warning(f"Skipping edge with invalid source/target: {edge_data}")
                        continue
                    
                    # Verify nodes exist before creating edge
                    source_node = ConceptNode.query.filter_by(id=source_id).first()
                    target_node = ConceptNode.query.filter_by(id=target_id).first()
                    
                    if not source_node or not target_node:
                        logging.warning(f"Skipping edge - nodes don't exist: source={source_id}, target={target_id}")
                        continue
                        
                    edge = ConceptEdge(
                        id=edge_id,
                        concept_session_id=concept_session.id,
                        source_node_id=source_id,
                        target_node_id=target_id,
                        edge_type=edge_data.get('type', 'relates_to')
                    )
                    db.session.add(edge)
                    edges_added += 1
                    logging.debug(f"Added edge: {source_id} -> {target_id} ({edge_data.get('type')})")
        
        # Update discourse type if changed
        if 'discourse_type' in concept_update:
            concept_session.discourse_type = concept_update['discourse_type']
            concept_session.updated_at = datetime.utcnow()
        
        # Commit all changes
        db.session.commit()
        
        logging.info(f"Successfully saved concept update for session_device {session_device_id}: "
                    f"{nodes_added} nodes, {edges_added} edges added")
        
        return jsonify({
            'status': 'success',
            'nodes_added': nodes_added,
            'edges_added': edges_added,
            'session_device_id': session_device_id,
            'concept_session_id': concept_session.id
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Failed to process concept update: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@concept_bp.route('/api/v1/concepts/<session_device_id>', methods=['GET'])
def get_concept_graph(session_device_id):
    """Get the concept graph for a session device"""
    try:
        # Parse to get the actual session_device_id
        db_session_device_id = parse_session_device_id(session_device_id)

        if db_session_device_id is None:
            logging.info(f"No session_device found for {session_device_id}")
            return jsonify({
                'nodes': [],
                'edges': [],
                'discourse_type': 'exploratory',
                'generation_status': 'pending'
            }), 200

        # Get the concept session
        concept_session = ConceptSession.query.filter_by(
            session_device_id=db_session_device_id
        ).first()

        if not concept_session:
            logging.info(f"No concept session found for session_device_id={db_session_device_id}")
            return jsonify({
                'nodes': [],
                'edges': [],
                'discourse_type': 'exploratory',
                'generation_status': 'pending'
            }), 200

        # Use the json() method from your model
        return jsonify(concept_session.json()), 200

    except Exception as e:
        logging.error(f"Failed to get concept graph: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    
@concept_bp.route('/api/v1/concepts/<session_device_id>/poll', methods=['GET'])
def poll_concepts(session_device_id):
    """Polling endpoint with timestamp for efficiency"""
    try:
        since = request.args.get('since', type=float, default=0)
        
        # Parse session_device_id
        db_session_device_id = parse_session_device_id(session_device_id)
        
        if db_session_device_id is None:
            return jsonify({'nodes': [], 'edges': []}), 200
        
        concept_session = ConceptSession.query.filter_by(
            session_device_id=db_session_device_id
        ).first()
        
        if not concept_session:
            return jsonify({'nodes': [], 'edges': []}), 200
        
        # Return only new items if timestamp provided
        if since > 0:
            # Filter nodes created after timestamp
            nodes = [n.json() for n in concept_session.nodes if n.timestamp and n.timestamp > since]
            edges = [e.json() for e in concept_session.edges]  # Edges might not have timestamps
            return jsonify({
                'nodes': nodes, 
                'edges': edges,
                'discourse_type': concept_session.discourse_type
            }), 200
        
        return jsonify(concept_session.json()), 200
        
    except Exception as e:
        logging.error(f"Polling error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@concept_bp.route('/api/v1/concepts/session/<int:session_id>/delete', methods=['DELETE'])
def delete_session_concepts(session_id):
    """Delete all concept data for a session"""
    try:
        from tables.session_device import SessionDevice
        
        # Find all session_devices for this session
        session_devices = SessionDevice.query.filter_by(
            session_id=session_id
        ).all()
        
        deleted_count = 0
        for session_device in session_devices:
            # Find concept sessions for each session_device
            concept_sessions = ConceptSession.query.filter_by(
                session_device_id=session_device.id
            ).all()
            
            for concept_session in concept_sessions:
                # The cascade delete should handle nodes and edges
                db.session.delete(concept_session)
                deleted_count += 1
        
        db.session.commit()
        logging.info(f"Deleted {deleted_count} concept sessions for session_id={session_id}")
        
        return jsonify({
            'status': 'success',
            'deleted_sessions': deleted_count
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Failed to delete concept sessions: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500




@concept_bp.route('/api/v1/concepts/broadcast', methods=['POST'])
def broadcast_concepts():
    """Endpoint to trigger WebSocket broadcast of concept updates"""
    try:
        from websocket_handler import broadcast_concept_update
        from app import socketio
        
        data = request.json
        session_device_id = data.get('session_device_id')
        concept_update = data.get('concept_update')
        
        if not session_device_id or not concept_update:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Broadcast to WebSocket clients
        broadcast_concept_update(socketio, session_device_id, concept_update)
        
        return jsonify({'success': True}), 200
    except Exception as e:
        logging.error(f"Broadcast endpoint error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500



@concept_bp.route('/api/v1/concepts/<session_device_id>/cluster', methods=['POST'])
def create_clusters(session_device_id):
    """Create clusters for a concept session (typically called after session ends)"""
    try:
        # Parse session_device_id
        db_session_device_id = parse_session_device_id(session_device_id)
        
        if db_session_device_id is None:
            return jsonify({'error': 'Session device not found'}), 404
        
        # Check if clusters already exist
        concept_session = ConceptSession.query.filter_by(
            session_device_id=db_session_device_id
        ).first()
        
        if not concept_session:
            return jsonify({'error': 'No concept session found'}), 404
        
        # Check for existing clusters
        existing_clusters = ConceptCluster.query.filter_by(
            concept_session_id=concept_session.id
        ).first()
        
        if existing_clusters:
            # Option to force re-clustering
            data = request.get_json() or {}
            if not data.get('force', False):
                return jsonify({
                    'message': 'Clusters already exist', 
                    'cluster_count': ConceptCluster.query.filter_by(
                        concept_session_id=concept_session.id
                    ).count()
                }), 200
            else:
                # Delete existing clusters
                ConceptCluster.query.filter_by(
                    concept_session_id=concept_session.id
                ).delete()
                db.session.commit()
        
        # Get clustering method from request
        data = request.get_json() or {}
        method = data.get('method', 'semantic')  # 'semantic' or 'time'
        
        if method == 'semantic':
            from concept_clustering_semantic import create_semantic_clusters
            cluster_ids = create_semantic_clusters(db_session_device_id)
        else:
            from concept_clustering import create_time_based_clusters
            time_window = data.get('time_window_minutes', 3)
            cluster_ids = create_time_based_clusters(db_session_device_id, time_window)

        # Trigger session-level RAG indexing after clustering completes
        session_indexed = _index_session_for_rag(db_session_device_id)

        return jsonify({
            'status': 'success',
            'method': method,
            'clusters_created': len(cluster_ids),
            'cluster_ids': cluster_ids,
            'session_indexed': session_indexed  # NEW: indicates if RAG index was updated
        }), 200
        
    except Exception as e:
        logging.error(f"Failed to create clusters: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@concept_bp.route('/api/v1/concepts/<session_device_id>/clusters', methods=['GET'])
def get_clusters(session_device_id):
    """Get clusters for a concept session"""
    try:
        from tables.concept_cluster import ConceptCluster
        
        # Parse session_device_id
        db_session_device_id = parse_session_device_id(session_device_id)
        
        if db_session_device_id is None:
            return jsonify({'clusters': []}), 200
        
        concept_session = ConceptSession.query.filter_by(
            session_device_id=db_session_device_id
        ).first()
        
        if not concept_session:
            return jsonify({'clusters': []}), 200
        
        # Get all clusters for this session
        clusters = ConceptCluster.query.filter_by(
            concept_session_id=concept_session.id
        ).order_by(ConceptCluster.cluster_order).all()
        
        # Get all edges for the session
        all_edges = ConceptEdge.query.filter_by(
            concept_session_id=concept_session.id
        ).all()
        
        # Build cluster data with edges
        cluster_data = []
        for cluster in clusters:
            cluster_json = cluster.json()
            
            # Find edges within this cluster
            cluster_node_ids = {node.id for node in cluster.nodes}
            cluster_edges = [
                edge.json() for edge in all_edges 
                if edge.source_node_id in cluster_node_ids and edge.target_node_id in cluster_node_ids
            ]
            cluster_json['edges'] = cluster_edges
            
            cluster_data.append(cluster_json)
        
        return jsonify({
            'clusters': cluster_data,
            'discourse_type': concept_session.discourse_type
        }), 200
        
    except Exception as e:
        logging.error(f"Failed to get clusters: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500