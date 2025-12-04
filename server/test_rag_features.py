import requests
import json

BASE_URL = "http://localhost:5000"

def search_topics():
    """Test different topic searches"""
    print("\n=== Topic Search Examples ===\n")
    
    topics = [
        ("high participation", "Find discussions with active participation"),
        ("analytical thinking", "Find analytical discussions"),
        ("emotional tone", "Find emotionally engaging discussions"),
        ("equilibrium", "Find chemistry-specific discussions"),
        ("collaboration", "Find collaborative moments")
    ]
    
    for query, description in topics:
        print(f"{description}:")
        response = requests.post(
            f"{BASE_URL}/api/v1/rag/search",
            json={"query": query, "n_results": 2}
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"  Found {results['total_found']} matches")
            
            for i, r in enumerate(results['results'][:2], 1):
                meta = r['metadata']
                print(f"  {i}. Session {meta['session_device_id']}, "
                      f"{meta['start_time']:.0f}s, "
                      f"{meta['speaker_count']} speakers")
                print(f"     Preview: {r['text'][:100]}...")
        print()

def find_patterns():
    """Search for collaboration patterns"""
    print("\n=== Pattern Search Examples ===\n")
    
    # Search with metric filters
    print("High emotional engagement discussions (emotional_tone > 50):")
    response = requests.post(
        f"{BASE_URL}/api/v1/rag/search",
        json={
            "query": "discussion",
            "n_results": 3,
            "filters": {"min_emotional_tone": 50}
        }
    )
    
    if response.status_code == 200:
        results = response.json()
        for r in results['results']:
            meta = r['metadata']
            print(f"  Session {meta['session_device_id']}: "
                  f"Emotional={meta['avg_emotional_tone']:.1f}, "
                  f"Analytic={meta['avg_analytic_thinking']:.1f}")

def similar_sessions():
    """Find similar discussion sessions"""
    print("\n=== Similar Sessions ===\n")
    
    # Use session 448 as reference
    reference_id = 448
    print(f"Finding sessions similar to session_device {reference_id}:")
    
    response = requests.get(
        f"{BASE_URL}/api/v1/rag/similar/{reference_id}",
        params={"n_results": 3}
    )
    
    if response.status_code == 200:
        results = response.json()
        if 'similar_sessions' in results:
            for session in results['similar_sessions']:
                print(f"  Session {session['session_device_id']}: "
                      f"similarity distance={session['avg_distance']:.3f}")
                print(f"    Matched {len(session['chunks'])} chunks")

def generate_insights_example():
    """Generate insights from search results"""
    print("\n=== AI-Generated Insights ===\n")
    
    # First search for a pattern
    search_response = requests.post(
        f"{BASE_URL}/api/v1/rag/search",
        json={
            "query": "effective collaboration strategies",
            "n_results": 5
        }
    )
    
    if search_response.status_code == 200:
        search_results = search_response.json()
        
        # Generate insights
        print("Generating insights about 'effective collaboration strategies'...")
        insights_response = requests.post(
            f"{BASE_URL}/api/v1/rag/insights",
            json={
                "query": "effective collaboration strategies",
                "search_results": search_results
            }
        )
        
        if insights_response.status_code == 200:
            insights = insights_response.json()
            print(f"\n{insights['insights']}")

if __name__ == "__main__":
    print("RAG Feature Demonstration")
    print("=" * 50)
    
    search_topics()
    find_patterns()
    similar_sessions()
    generate_insights_example()
    
    print("\n" + "=" * 50)
    print("RAG system is ready for use!")