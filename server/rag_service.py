import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize RAG service with ChromaDB and OpenAI"""
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize OpenAI
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Create embedding function
        self.embedding_function = OpenAIEmbeddingFunction(
            api_key=self.openai_api_key,
            model_name="text-embedding-3-small"
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="discussion_chunks",
            embedding_function=self.embedding_function,
            metadata={"description": "30-second discussion chunks with metadata"}
        )
        
        logger.info(f"RAG Service initialized with collection size: {self.collection.count()}")
    
    def chunk_transcripts(self, transcripts: List, window_seconds: int = 30) -> List[Dict]:
        """
        Chunk transcripts into fixed time windows
        
        Args:
            transcripts: List of transcript objects from database
            window_seconds: Size of each chunk in seconds
            
        Returns:
            List of chunks with metadata
        """
        if not transcripts:
            return []
        
        chunks = []
        current_chunk_transcripts = []
        current_chunk_start = 0
        
        for transcript in transcripts:
            # Check if transcript belongs to current window
            if transcript.start_time < current_chunk_start + window_seconds:
                current_chunk_transcripts.append(transcript)
            else:
                # Save current chunk if it has content
                if current_chunk_transcripts:
                    chunks.append(self._create_chunk(
                        current_chunk_transcripts, 
                        current_chunk_start,
                        current_chunk_start + window_seconds
                    ))
                
                # Start new chunk
                current_chunk_transcripts = [transcript]
                current_chunk_start = (transcript.start_time // window_seconds) * window_seconds
        
        # Don't forget the last chunk
        if current_chunk_transcripts:
            chunks.append(self._create_chunk(
                current_chunk_transcripts,
                current_chunk_start,
                current_chunk_start + window_seconds
            ))
        
        return chunks
    
    def _create_chunk(self, transcripts: List, start_time: float, end_time: float) -> Dict:
        """Create a chunk with text and metadata"""
        
        # Combine transcript texts
        texts = []
        speakers = set()
        total_emotional = 0
        total_analytic = 0
        total_clout = 0
        total_authenticity = 0
        total_certainty = 0
        count = 0
        
        for t in transcripts:
            # Add speaker prefix if available
            if hasattr(t, 'speaker_tag') and t.speaker_tag:
                texts.append(f"{t.speaker_tag}: {t.transcript}")
                speakers.add(t.speaker_tag)
            else:
                texts.append(t.transcript)
            
            # Aggregate metrics
            if hasattr(t, 'emotional_tone_value'):
                total_emotional += t.emotional_tone_value or 0
                total_analytic += t.analytic_thinking_value or 0
                total_clout += t.clout_value or 0
                total_authenticity += t.authenticity_value or 0
                total_certainty += t.certainty_value or 0
                count += 1
        
        # Calculate averages
        if count > 0:
            avg_emotional = round(total_emotional / count, 2)
            avg_analytic = round(total_analytic / count, 2)
            avg_clout = round(total_clout / count, 2)
            avg_authenticity = round(total_authenticity / count, 2)
            avg_certainty = round(total_certainty / count, 2)
        else:
            avg_emotional = avg_analytic = avg_clout = avg_authenticity = avg_certainty = 0
        
        chunk_text = " ".join(texts)
        
        return {
            "text": chunk_text,
            "start_time": start_time,
            "end_time": end_time,
            "speaker_count": len(speakers),
            "speakers": list(speakers),
            "transcript_count": len(transcripts),
            "avg_emotional_tone": avg_emotional,
            "avg_analytic_thinking": avg_analytic,
            "avg_clout": avg_clout,
            "avg_authenticity": avg_authenticity,
            "avg_certainty": avg_certainty,
            "session_device_id": transcripts[0].session_device_id if transcripts else None
        }
    
    def index_session_chunks(self, session_device_id: int, transcripts: List) -> int:
        """
        Index all chunks from a session
        
        Returns:
            Number of chunks indexed
        """
        chunks = self.chunk_transcripts(transcripts)
        
        if not chunks:
            logger.warning(f"No chunks created for session_device {session_device_id}")
            return 0
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"sd{session_device_id}_chunk{i}_{int(chunk['start_time'])}"
            
            documents.append(chunk["text"])
            ids.append(chunk_id)
            
            # Prepare metadata (ChromaDB requires serializable types)
            metadata = {
                "session_device_id": session_device_id,
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "speaker_count": chunk["speaker_count"],
                "speakers": json.dumps(chunk["speakers"]),  # Convert list to string
                "transcript_count": chunk["transcript_count"],
                "avg_emotional_tone": chunk["avg_emotional_tone"],
                "avg_analytic_thinking": chunk["avg_analytic_thinking"],
                "avg_clout": chunk["avg_clout"],
                "avg_authenticity": chunk["avg_authenticity"],
                "avg_certainty": chunk["avg_certainty"]
            }
            metadatas.append(metadata)
        
        # Add to ChromaDB
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Indexed {len(chunks)} chunks for session_device {session_device_id}")
            return len(chunks)
        except Exception as e:
            logger.error(f"Error indexing session_device {session_device_id}: {e}")
            return 0
    
    def search(self, 
              query: str, 
              n_results: int = 5,
              session_device_id: Optional[int] = None,
              session_device_ids: Optional[List[int]] = None,
              min_emotional_tone: Optional[float] = None,
              min_analytic_thinking: Optional[float] = None) -> Dict:
        """
        Search for relevant discussion chunks
        
        Args:
            query: Search query text
            n_results: Number of results to return
            session_device_id: Filter by specific session_device
            min_emotional_tone: Filter by minimum emotional tone
            min_analytic_thinking: Filter by minimum analytic thinking
            
        Returns:
            Search results with documents and metadata
        """
        
        # Build where clause for filtering
        where = {}
        if session_device_id:
            where["session_device_id"] = session_device_id
        elif session_device_ids:  # Use list filter if provided
            where["session_device_id"] = {"$in": session_device_ids}
        if min_emotional_tone:
            where["avg_emotional_tone"] = {"$gte": min_emotional_tone}
        if min_analytic_thinking:
            where["avg_analytic_thinking"] = {"$gte": min_analytic_thinking}
        
        # Perform search
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where if where else None
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    "chunk_id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "distance": results['distances'][0][i],
                    "metadata": results['metadatas'][0][i]
                }
                # Parse speakers back to list
                if 'speakers' in result['metadata']:
                    result['metadata']['speakers'] = json.loads(result['metadata']['speakers'])
                formatted_results.append(result)
            
            return {
                "query": query,
                "results": formatted_results,
                "total_found": len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                "query": query,
                "results": [],
                "total_found": 0,
                "error": str(e)
            }
    
    def find_similar_discussions(self, session_device_id: int, n_results: int = 5) -> Dict:
        """
        Find discussions similar to a given session
        
        Args:
            session_device_id: Reference session_device ID
            n_results: Number of similar discussions to return
            
        Returns:
            Similar discussion chunks from other sessions
        """
        
        # Get chunks from the reference session
        reference_chunks = self.collection.get(
            where={"session_device_id": session_device_id},
            limit=5  # Use first 5 chunks as reference
        )
        
        if not reference_chunks['ids']:
            return {
                "reference_session_device_id": session_device_id,
                "results": [],
                "message": "No chunks found for reference session"
            }
        
        # Combine reference texts
        reference_text = " ".join(reference_chunks['documents'])
        
        # Search for similar, excluding the reference session
        results = self.collection.query(
            query_texts=[reference_text[:1000]],  # Limit text length
            n_results=n_results * 2,  # Get more to filter
            where={"session_device_id": {"$ne": session_device_id}}
        )
        
        # Group by session and format
        session_groups = {}
        for i in range(len(results['ids'][0])):
            sd_id = results['metadatas'][0][i]['session_device_id']
            if sd_id not in session_groups:
                session_groups[sd_id] = {
                    "session_device_id": sd_id,
                    "chunks": [],
                    "avg_distance": 0
                }
            session_groups[sd_id]["chunks"].append({
                "text": results['documents'][0][i],
                "distance": results['distances'][0][i],
                "metadata": results['metadatas'][0][i]
            })
        
        # Calculate average distances and sort
        for sd_id in session_groups:
            chunks = session_groups[sd_id]["chunks"]
            session_groups[sd_id]["avg_distance"] = sum(c["distance"] for c in chunks) / len(chunks)
        
        sorted_sessions = sorted(session_groups.values(), key=lambda x: x["avg_distance"])[:n_results]
        
        return {
            "reference_session_device_id": session_device_id,
            "similar_sessions": sorted_sessions
        }
    
    def generate_insights(self, query: str, search_results: Dict) -> str:
        """
        Generate insights based on search results using GPT-4
        
        Args:
            query: Original search query
            search_results: Results from search()
            
        Returns:
            Generated insights text
        """
        
        if not search_results['results']:
            return "No relevant discussions found for generating insights."
        
        # Prepare context from search results
        context_parts = []
        for i, result in enumerate(search_results['results'][:5], 1):
            metadata = result['metadata']
            context_parts.append(
                f"Excerpt {i} (Session {metadata['session_device_id']}, "
                f"{metadata['start_time']:.0f}-{metadata['end_time']:.0f}s, "
                f"Speakers: {metadata.get('speakers', 'Unknown')}):\n"
                f"{result['text']}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Generate insights with GPT-4
        prompt = f"""Based on these discussion excerpts related to "{query}", provide insights about:
1. Common patterns or themes
2. Collaboration dynamics observed
3. Key discussion characteristics
4. Actionable recommendations

Context:
{context}

Provide specific, evidence-based insights with references to the excerpts."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Using mini for cost efficiency
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing discussion patterns and collaboration dynamics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return f"Error generating insights: {str(e)}"
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the indexed collection"""
        
        count = self.collection.count()
        
        # Get sample to understand the data
        if count > 0:
            sample = self.collection.get(limit=1)
            return {
                "total_chunks": count,
                "collection_name": "discussion_chunks",
                "sample_metadata_keys": list(sample['metadatas'][0].keys()) if sample['metadatas'] else []
            }
        
        return {
            "total_chunks": 0,
            "collection_name": "discussion_chunks",
            "sample_metadata_keys": []
        }