"""
Post-discussion concept map generation service.

This service generates concept maps after a discussion session ends,
allowing the LLM to have access to the full transcript context for better quality.
"""
import logging
import json
import database
from datetime import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv
from tables.concept_session import ConceptSession
from tables.concept_node import ConceptNode
from tables.concept_edge import ConceptEdge
from app import db

load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None


def generate_concepts_for_session_device(session_device_id):
    """
    Generate concept map from complete discussion transcripts.
    Called as a background job when a session ends.

    Args:
        session_device_id: ID of the session device to process

    Returns:
        ConceptSession object if successful, None if failed
    """
    if not client:
        logging.error(f"Cannot generate concepts - OpenAI client not initialized")
        return None

    logging.info(f"Starting post-discussion concept generation for session_device {session_device_id}")

    try:
        # Find or create ConceptSession
        concept_session = ConceptSession.query.filter_by(
            session_device_id=session_device_id
        ).first()

        if not concept_session:
            concept_session = ConceptSession(session_device_id=session_device_id)
            db.session.add(concept_session)
            db.session.commit()

        # Update status to processing
        concept_session.generation_status = 'processing'
        db.session.commit()

        # Get ALL transcripts for this session device
        transcripts = database.get_transcripts(session_device_id=session_device_id)

        if not transcripts:
            logging.info(f"No transcripts found for session_device {session_device_id}")
            concept_session.generation_status = 'completed'
            concept_session.generated_at = datetime.utcnow()
            db.session.commit()
            return concept_session

        # Prepare full transcript text with line numbers
        transcript_text, line_to_timestamp = prepare_transcript_for_concepts(transcripts)

        # Get speaker aliases for the response
        speaker_aliases = get_speaker_aliases(transcripts)

        # Call LLM to extract concepts (pass line mapping for timestamp resolution)
        concepts_result = extract_concepts_from_full_transcript(transcript_text, line_to_timestamp)

        if not concepts_result:
            logging.error(f"Failed to extract concepts for session_device {session_device_id}")
            concept_session.generation_status = 'failed'
            concept_session.generation_error = 'LLM extraction failed'
            db.session.commit()
            return None

        # Clear existing nodes and edges (in case of re-generation)
        ConceptNode.query.filter_by(concept_session_id=concept_session.id).delete()
        ConceptEdge.query.filter_by(concept_session_id=concept_session.id).delete()
        db.session.commit()

        # Store nodes
        nodes = concepts_result.get('nodes', [])
        node_id_map = {}  # Map from index to actual node ID

        for i, node_data in enumerate(nodes):
            node_id = f"node_{session_device_id}_{i}"
            speaker_id = node_data.get('speaker')

            # Try to get speaker ID if it's a number or alias
            resolved_speaker_id = None
            if speaker_id:
                if isinstance(speaker_id, int):
                    resolved_speaker_id = speaker_id
                elif str(speaker_id).isdigit():
                    resolved_speaker_id = int(speaker_id)

            node = ConceptNode(
                id=node_id,
                concept_session_id=concept_session.id,
                text=node_data.get('text', ''),
                node_type=node_data.get('type', 'concept'),
                speaker_id=resolved_speaker_id,
                timestamp=node_data.get('timestamp', 0)
            )
            db.session.add(node)
            node_id_map[i] = node_id

        db.session.commit()

        # Store edges
        edges = concepts_result.get('edges', [])
        for i, edge_data in enumerate(edges):
            source_idx = edge_data.get('source')
            target_idx = edge_data.get('target')

            # Validate indices
            if source_idx not in node_id_map or target_idx not in node_id_map:
                logging.warning(f"Skipping edge with invalid indices: {source_idx} -> {target_idx}")
                continue

            edge_id = f"edge_{session_device_id}_{i}"
            edge = ConceptEdge(
                id=edge_id,
                concept_session_id=concept_session.id,
                source_node_id=node_id_map[source_idx],
                target_node_id=node_id_map[target_idx],
                edge_type=edge_data.get('type', 'relates_to')
            )
            db.session.add(edge)

        # Update discourse type
        concept_session.discourse_type = concepts_result.get('discourse_type', 'exploratory')
        concept_session.generation_status = 'completed'
        concept_session.generated_at = datetime.utcnow()
        db.session.commit()

        logging.info(f"Successfully generated {len(nodes)} concepts and {len(edges)} edges for session_device {session_device_id}")

        # Now trigger clustering
        try:
            from concept_clustering_semantic import create_semantic_clusters
            cluster_ids = create_semantic_clusters(session_device_id)
            if cluster_ids:
                logging.info(f"Created {len(cluster_ids)} clusters for session_device {session_device_id}")
        except Exception as e:
            logging.error(f"Failed to create clusters: {e}")

        return concept_session

    except Exception as e:
        logging.error(f"Error generating concepts for session_device {session_device_id}: {str(e)}", exc_info=True)

        # Update status to failed
        try:
            concept_session = ConceptSession.query.filter_by(
                session_device_id=session_device_id
            ).first()
            if concept_session:
                concept_session.generation_status = 'failed'
                concept_session.generation_error = str(e)
                db.session.commit()
        except:
            pass

        return None


def get_speaker_aliases(transcripts):
    """Get a mapping of speaker_id to alias for all speakers in transcripts."""
    speaker_ids = set()
    for t in transcripts:
        if t.speaker_id:
            speaker_ids.add(t.speaker_id)

    speaker_map = {}
    for speaker_id in speaker_ids:
        speaker = database.get_speakers(id=speaker_id)
        if speaker:
            speaker_map[speaker_id] = speaker.get_alias()
        else:
            speaker_map[speaker_id] = f"Speaker {speaker_id}"

    return speaker_map


def prepare_transcript_for_concepts(transcripts):
    """
    Prepare full transcript text for concept extraction.

    Args:
        transcripts: List of transcript objects

    Returns:
        Tuple of (formatted_string, line_to_timestamp_map)
        - formatted_string: Text with line numbers and speaker labels
        - line_to_timestamp_map: Dict mapping line numbers to start_time values
    """
    speaker_aliases = get_speaker_aliases(transcripts)

    # Format transcripts with line numbers for accurate reference
    transcript_lines = []
    line_to_timestamp = {}

    for i, t in enumerate(transcripts):
        line_num = i + 1  # 1-indexed for readability

        if t.speaker_id and t.speaker_id in speaker_aliases:
            speaker = speaker_aliases[t.speaker_id]
        elif t.speaker_tag:
            speaker = f"Speaker {t.speaker_tag}"
        else:
            speaker = "Unknown"

        time_min = int(t.start_time // 60)
        time_sec = int(t.start_time % 60)

        # Include line number for accurate referencing
        transcript_lines.append(f"L{line_num} [{time_min}:{time_sec:02d}] {speaker}: {t.transcript}")

        # Map line number to actual timestamp
        line_to_timestamp[line_num] = t.start_time

    return "\n".join(transcript_lines), line_to_timestamp


def extract_concepts_from_full_transcript(transcript_text, line_to_timestamp=None):
    """
    Extract concepts and relationships from the complete discussion transcript.

    Args:
        transcript_text: Full formatted transcript with line numbers (L1, L2, etc.)
        line_to_timestamp: Dict mapping line numbers to actual timestamps

    Returns:
        Dict with nodes, edges, and discourse_type
    """
    if line_to_timestamp is None:
        line_to_timestamp = {}

    # GPT-4o supports 128K tokens. 400K chars (~100K tokens) is safe for full discussions
    max_chars = 400000  # ~100k tokens - plenty of room for most discussions
    if len(transcript_text) > max_chars:
        logging.warning(f"Transcript very long ({len(transcript_text)} chars), truncating to {max_chars}")
        transcript_text = transcript_text[:max_chars] + "\n[... transcript truncated ...]"

    prompt = f"""Analyze this discussion and extract an INTERCONNECTED knowledge graph.

CRITICAL: Create a GRAPH, not a chain. Concepts should have multiple connections.
- Link concepts by THEME, not just by sequence
- Find connections between ideas mentioned at DIFFERENT times in the discussion
- The same topic discussed early and late should be connected
- Aim for concepts to have 2-4 relationships each, not just 1

DISCUSSION TRANSCRIPT (each line starts with L# for line reference):
{transcript_text}

EXTRACTION GUIDELINES:
1. First pass: Identify ALL key concepts throughout the discussion
2. Second pass: Find THEMATIC relationships (same topic, contrasting views, supporting evidence)
3. Connect concepts that share themes even if mentioned far apart
4. Avoid purely sequential chains - if A→B→C, also look for A→C or other cross-links
5. Look for recurring themes and connect all instances

CONCEPT TYPES:
- idea: Main concepts and claims
- question: Questions asked (preserve question form)
- example: Concrete examples
- problem: Problems or challenges identified
- solution: Proposed solutions or approaches
- goal: Stated objectives or goals
- uncertainty: Expressed doubts or unknowns
- conclusion: Final decisions or conclusions reached
- action: Action items or next steps

RELATIONSHIP TYPES (prioritize non-sequential connections):
- relates_to: Thematically connected concepts
- similar_to: Concepts expressing similar ideas
- contrasts_with: Opposing or alternative viewpoints
- supports: Evidence or agreement
- challenges: Disagreements or counterpoints
- elaborates: Adding detail (can skip intermediate concepts)
- answers: Response to a question (may be distant in transcript)
- exemplifies: Concrete example of abstract concept
- synthesizes: Combining multiple ideas
- builds_on: Only when genuinely developing an idea (not just "next in sequence")

IMPORTANT REQUIREMENTS:
- Each concept should connect to 2-4 other concepts
- At least 30% of edges should connect NON-ADJACENT concepts (different parts of discussion)
- Look for the SAME THEME appearing multiple times and connect those instances
- Extract 15-40 concepts depending on discussion length
- For "source_line", use the LINE NUMBER (e.g., 5 for L5) where the concept is primarily mentioned

Return a JSON object:
{{
    "nodes": [
        {{"text": "concept text (3-20 words)", "type": "type", "speaker": "speaker_id_or_name", "source_line": line_number}}
    ],
    "edges": [
        {{"source": node_index, "target": node_index, "type": "relationship_type"}}
    ],
    "discourse_type": "exploratory|problem_solving|analytical|planning|mixed",
    "summary": "1-2 sentence summary of the discussion"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert knowledge graph extractor specializing in academic discussions.
Your goal is to create an INTERCONNECTED knowledge graph, not a linear chain.

KEY PRINCIPLES:
- Find THEMATIC connections between concepts, regardless of when they were mentioned
- Link related ideas even if they appear at different points in the discussion
- Create a web of relationships, not a timeline
- Identify recurring themes and connect all instances
- Each concept should have multiple connections (2-4 edges)
- Prioritize cross-referencing over sequential links"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=4000
        )

        result = json.loads(response.choices[0].message.content)

        # Validate and clean the result
        nodes = result.get('nodes', [])
        edges = result.get('edges', [])

        valid_nodes = []
        for node in nodes:
            if node.get('text'):
                # Convert source_line to actual timestamp using the mapping
                source_line = node.get('source_line')
                timestamp = 0

                if source_line and isinstance(source_line, int) and source_line in line_to_timestamp:
                    timestamp = line_to_timestamp[source_line]
                elif node.get('timestamp'):
                    # Fallback to old format for backwards compatibility
                    timestamp = node.get('timestamp', 0)

                valid_nodes.append({
                    'text': node.get('text', ''),
                    'type': node.get('type', 'concept'),
                    'speaker': node.get('speaker', 'Unknown'),
                    'timestamp': timestamp,
                    'source_line': source_line  # Keep reference for debugging
                })

        valid_edges = []
        for edge in edges:
            if 'source' in edge and 'target' in edge:
                source = edge['source']
                target = edge['target']
                # Validate indices
                if isinstance(source, int) and isinstance(target, int):
                    if 0 <= source < len(valid_nodes) and 0 <= target < len(valid_nodes):
                        valid_edges.append({
                            'source': source,
                            'target': target,
                            'type': edge.get('type', 'relates_to')
                        })

        logging.info(f"Extracted {len(valid_nodes)} nodes and {len(valid_edges)} edges from full transcript")

        return {
            'nodes': valid_nodes,
            'edges': valid_edges,
            'discourse_type': result.get('discourse_type', 'exploratory'),
            'summary': result.get('summary', '')
        }

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse LLM response as JSON: {e}")
        return None
    except Exception as e:
        logging.error(f"Error calling OpenAI API for concept extraction: {e}")
        return None
