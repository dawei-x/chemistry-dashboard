import requests
import json

BASE_URL = "http://localhost:5002"

def test_search():
    """Test basic search"""
    print("\n=== Testing Search ===")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/rag/search",
        json={
            "query": "collaboration and participation",
            "n_results": 3
        }
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"Found {results['total_found']} results")
        for i, result in enumerate(results['results'], 1):
            print(f"\nResult {i}:")
            print(f"  Session Device: {result['metadata']['session_device_id']}")
            print(f"  Time: {result['metadata']['start_time']:.0f}-{result['metadata']['end_time']:.0f}s")
            print(f"  Text preview: {result['text'][:100]}...")
            print(f"  Distance: {result['distance']:.3f}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_similar():
    """Test finding similar discussions"""
    print("\n=== Testing Similar Discussions ===")
    
    # Use a real session_device_id
    session_device_id = 448  # recent session
    
    response = requests.get(
        f"{BASE_URL}/api/v1/rag/similar/{session_device_id}",
        params={"n_results": 3}
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"Finding sessions similar to {results['reference_session_device_id']}")
        
        for session in results['similar_sessions']:
            print(f"\n  Session Device {session['session_device_id']}:")
            print(f"    Average distance: {session['avg_distance']:.3f}")
            print(f"    Number of matching chunks: {len(session['chunks'])}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_insights():
    """Test insight generation"""
    print("\n=== Testing Insights Generation ===")
    
    # First do a search
    search_response = requests.post(
        f"{BASE_URL}/api/v1/rag/search",
        json={
            "query": "effective discussion strategies",
            "n_results": 5
        }
    )
    
    if search_response.status_code == 200:
        search_results = search_response.json()
        
        # Generate insights
        insights_response = requests.post(
            f"{BASE_URL}/api/v1/rag/insights",
            json={
                "query": "effective discussion strategies",
                "search_results": search_results
            }
        )
        
        if insights_response.status_code == 200:
            insights = insights_response.json()
            print(f"Query: {insights['query']}")
            print(f"\nInsights:\n{insights['insights']}")
        else:
            print(f"Insights Error: {insights_response.status_code}")
    else:
        print(f"Search Error: {search_response.status_code}")

if __name__ == "__main__":
    print("Testing RAG API endpoints...")
    
    test_search()
    test_similar() 
    test_insights()
    
    print("\n=== All tests complete ===")