# server/rag_query_parser.py - With Auto-Insights

import re
import json
import logging
from typing import Dict, List, Optional
from rag_service import RAGService
import database

logger = logging.getLogger(__name__)

class QueryParser:
    def __init__(self):
        self.rag = RAGService()
        
    def parse_and_execute(self, query: str, user_session_devices: Optional[List[int]] = None) -> Dict:
        """Parse natural language query and execute appropriate search"""
        
        query_lower = query.lower()
        
        # Check for specific query types first
        if self._is_comparative(query_lower):
            return self._handle_comparative(query, user_session_devices)
        elif self._is_similarity(query_lower):
            return self._handle_similarity_search(query, user_session_devices)
        elif self._is_temporal(query_lower):
            return self._handle_temporal_analysis(query, user_session_devices)
        else:
            # Use intelligent parsing for everything else
            return self._handle_intelligent_search(query, user_session_devices)
    
    def _is_comparative(self, query: str) -> bool:
        """Check if query is asking for comparison"""
        return any(word in query for word in ['compare', 'versus', 'vs', 'difference between', 'contrast'])
    
    def _is_similarity(self, query: str) -> bool:
        """Check if query is asking for similar sessions"""
        return any(phrase in query for phrase in ['similar to', 'like session', 'resembles', 'same as'])
    
    def _is_temporal(self, query: str) -> bool:
        """Check if query is asking for temporal analysis"""
        return any(word in query for word in ['timeline', 'over time', 'progression', 'evolution', 'throughout'])
    
    def _is_analytical_query(self, query: str) -> bool:
        """Check if query is asking for analysis (auto-trigger insights)"""
        query_lower = query.lower()
        analytical_keywords = ['why', 'how', 'what patterns', 'what makes', 'analyze', 'explain', 
                               'insights', 'understand', 'reason', 'cause', 'leads to', 'results in']
        return any(keyword in query_lower for keyword in analytical_keywords)
    
    def _handle_intelligent_search(self, query: str, user_devices: Optional[List[int]] = None) -> Dict:
        """Use GPT to understand query intent and extract parameters"""
        
        # Check if this is an analytical query that needs auto-insights
        auto_generate_insights = self._is_analytical_query(query)
        
        # First try GPT parsing
        try:
            from openai import OpenAI
            client = OpenAI()
            
            prompt = f"""Parse this discussion analytics query: "{query}"
            
            Extract:
            1. search_terms: Key terms to search for (remove filler words)
            2. intent: One of [topic_search, pattern_analysis, insight_generation]
            3. temporal: Any time references
            
            Return as JSON. Example:
            {{"search_terms": "collaboration", "intent": "pattern_analysis", "temporal": null}}
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            parsed = json.loads(response.choices[0].message.content)
            logger.info(f"GPT parsed query: {parsed}")
            
            # Build filters (no metric filtering - only session filtering)
            filters = {}
            if user_devices:
                filters['session_device_ids'] = user_devices
            
            # Execute search
            search_results = self.rag.search(
                query=parsed.get('search_terms', query),
                n_results=5,
                **filters
            )
            
            logger.info(f"Search returned {search_results.get('total_found', 0)} results")
            
            # Auto-generate insights if analytical query OR if GPT detected insight intent
            if auto_generate_insights or parsed.get('intent') == 'insight_generation':
                logger.info(f"Auto-generating insights for analytical query")
                insights = self.rag.generate_insights(query, search_results)
                return {
                    'type': 'insights',
                    'query': query,
                    'parsed': parsed,
                    'insights': insights,
                    'evidence': search_results
                }
            
            # Otherwise return as simple retrieval
            return {
                'type': parsed.get('intent', 'topic_search'),
                'query': query,
                'parsed': parsed,
                'results': search_results
            }
            
        except Exception as e:
            logger.warning(f"GPT parsing failed, using fallback: {e}")
            return self._fallback_search(query, user_devices, auto_generate_insights)
    
    def _fallback_search(self, query: str, user_devices: Optional[List[int]] = None, 
                         auto_generate_insights: bool = False) -> Dict:
        """Fallback without GPT - use simple heuristics"""
        
        logger.info(f"Using fallback search for: {query}")
        
        # Remove common filler words
        stop_words = {'show', 'me', 'find', 'search', 'for', 'the', 'a', 'an', 'about', 'what', 'when', 'where'}
        words = query.lower().split()
        clean_query = ' '.join([w for w in words if w not in stop_words])
        
        # Build filters (no metric filtering - only session filtering)
        filters = {}
        if user_devices:
            filters['session_device_ids'] = user_devices
        
        results = self.rag.search(
            query=clean_query if clean_query else query,
            n_results=5,
            **filters
        )
        
        logger.info(f"Fallback search returned {results.get('total_found', 0)} results")
        
        # Auto-generate insights if needed
        if auto_generate_insights:
            logger.info(f"Auto-generating insights in fallback")
            insights = self.rag.generate_insights(query, results)
            return {
                'type': 'insights',
                'query': query,
                'interpreted_as': clean_query,
                'insights': insights,
                'evidence': results
            }
        
        return {
            'type': 'topic_search',
            'query': query,
            'interpreted_as': clean_query,
            'results': results
        }
    
    def _handle_comparative(self, query: str, user_devices: Optional[List[int]] = None) -> Dict:
        """Handle comparative queries between sessions"""
        
        logger.info(f"Handling comparative query: {query}")
        
        # Extract session/device IDs
        session_pattern = r'session\s*(\d+)'
        device_pattern = r'device\s*(\d+)'
        
        session_ids = re.findall(session_pattern, query.lower())
        device_ids = re.findall(device_pattern, query.lower())
        
        if len(session_ids) + len(device_ids) < 2:
            return {
                'type': 'error',
                'message': 'Please specify at least two sessions or devices to compare',
                'example': 'Compare device 448 and device 447'
            }
        
        # Collect devices to compare
        devices_to_compare = []
        
        for sid in session_ids:
            session_devices = database.get_session_devices(session_id=int(sid))
            devices_to_compare.extend([(sd.id, f"Session {sid} - {sd.name}") for sd in session_devices])
        
        for did in device_ids:
            devices_to_compare.append((int(did), f"Device {did}"))
        
        # Get metrics for each
        comparisons = {}
        for device_id, label in devices_to_compare:
            chunks = self.rag.collection.get(
                where={'session_device_id': device_id},
                limit=100
            )
            
            if chunks['metadatas']:
                metas = chunks['metadatas']
                comparisons[label] = {
                    'device_id': device_id,
                    'metrics': {
                        'avg_emotional_tone': round(sum(m.get('avg_emotional_tone', 0) for m in metas) / len(metas), 2),
                        'avg_analytic_thinking': round(sum(m.get('avg_analytic_thinking', 0) for m in metas) / len(metas), 2),
                        'avg_clout': round(sum(m.get('avg_clout', 0) for m in metas) / len(metas), 2),
                        'avg_authenticity': round(sum(m.get('avg_authenticity', 0) for m in metas) / len(metas), 2),
                        'avg_certainty': round(sum(m.get('avg_certainty', 0) for m in metas) / len(metas), 2)
                    },
                    'total_chunks': len(metas),
                    'unique_speakers': len(set(str(m.get('speakers', '[]')) for m in metas))
                }
        
        logger.info(f"Comparison completed for {len(comparisons)} devices")
        
        return {
            'type': 'comparative',
            'query': query,
            'comparisons': comparisons
        }
    
    def _handle_similarity_search(self, query: str, user_devices: Optional[List[int]] = None) -> Dict:
        """Find sessions similar to a reference"""
        
        logger.info(f"Handling similarity query: {query}")
        
        # Extract reference ID
        device_match = re.search(r'device\s*(\d+)', query.lower())
        session_match = re.search(r'session\s*(\d+)', query.lower())
        
        reference_id = None
        if device_match:
            reference_id = int(device_match.group(1))
        elif session_match:
            sid = int(session_match.group(1))
            devices = database.get_session_devices(session_id=sid)
            if devices:
                reference_id = devices[0].id
        
        if not reference_id:
            if user_devices and len(user_devices) > 0:
                reference_id = user_devices[0]
            else:
                return {
                    'type': 'error',
                    'message': 'Please specify a session or device to find similar discussions'
                }
        
        results = self.rag.find_similar_discussions(reference_id, n_results=5)
        
        logger.info(f"Found similar sessions")
        
        return {
            'type': 'similar',
            'query': query,
            'reference_device': reference_id,
            'results': results
        }
    
    def _handle_temporal_analysis(self, query: str, user_devices: Optional[List[int]] = None) -> Dict:
        """Analyze patterns over time for a session"""
        
        logger.info(f"Handling temporal query: {query}")
        
        # Extract session/device ID
        device_match = re.search(r'device\s*(\d+)', query.lower())
        session_match = re.search(r'session\s*(\d+)', query.lower())
        
        device_id = None
        if device_match:
            device_id = int(device_match.group(1))
        elif session_match:
            sid = int(session_match.group(1))
            devices = database.get_session_devices(session_id=sid)
            if devices:
                device_id = devices[0].id
        elif user_devices and len(user_devices) > 0:
            device_id = user_devices[0]
        
        if not device_id:
            return {
                'type': 'error',
                'message': 'Please specify a session or device for temporal analysis'
            }
        
        # Get all chunks in order
        chunks = self.rag.collection.get(
            where={'session_device_id': device_id},
            limit=1000
        )
        
        if not chunks['metadatas']:
            return {
                'type': 'error',
                'message': f'No data found for device {device_id}'
            }
        
        # Create timeline
        timeline_data = []
        for i, meta in enumerate(chunks['metadatas']):
            timeline_data.append({
                'index': i,
                'time': meta.get('start_time', 0),
                'metrics': {
                    'emotional_tone': meta.get('avg_emotional_tone', 0),
                    'analytic_thinking': meta.get('avg_analytic_thinking', 0),
                    'clout': meta.get('avg_clout', 0),
                    'authenticity': meta.get('avg_authenticity', 0),
                    'certainty': meta.get('avg_certainty', 0)
                },
                'speakers': meta.get('speaker_count', 0)
            })
        
        # Sort by time
        timeline_data.sort(key=lambda x: x['time'])
        
        logger.info(f"Temporal analysis completed with {len(timeline_data)} data points")
        
        return {
            'type': 'temporal',
            'query': query,
            'device_id': device_id,
            'timeline': timeline_data,
            'summary': {
                'total_duration': timeline_data[-1]['time'] if timeline_data else 0,
                'total_chunks': len(timeline_data)
            }
        }
    
    def get_suggested_queries(self, context: Optional[Dict] = None) -> List[Dict]:
        """Get contextual suggested queries"""
        
        base_suggestions = [
            {'category': 'Discovery', 'queries': [
                'Show me high engagement discussions',
                'Find analytical thinking moments',
                'Search for collaborative problem solving'
            ]},
            {'category': 'Analysis', 'queries': [
                'What patterns lead to productive discussions?',
                'Why does engagement vary across sessions?',
                'How do successful collaborations differ?'
            ]},
            {'category': 'Comparison', 'queries': [
                'Compare device 448 and device 447',
                'Show differences between sessions'
            ]}
        ]
        
        if context and context.get('session_device_id'):
            device_id = context['session_device_id']
            base_suggestions.append({
                'category': 'Current Session',
                'queries': [
                    f'Find sessions similar to device {device_id}',
                    f'Analyze device {device_id} progression over time',
                    f'What made device {device_id} unique?'
                ]
            })
        
        return base_suggestions