import requests
import json

BASE_URL = "http://localhost:5002"

def test_stats():
    """Test that RAG is integrated and working"""
    print("Testing RAG stats endpoint...")
    response = requests.get(f"{BASE_URL}/api/v1/rag/stats")
    if response.status_code == 200:
        stats = response.json()
        print(f"✓ RAG system online with {stats['total_chunks']} chunks indexed")
        return True
    else:
        print(f"✗ Error: {response.status_code}")
        return False

def test_search_topics():
    """Test searching for specific topics"""
    print("\nTesting topic search...")
    
    queries = [
        "collaboration patterns",
        "analytical thinking",
        "emotional discussion",
        "participation"
    ]
    
    for query in queries:
        response = requests.post(
            f"{BASE_URL}/api/v1/rag/search",
            json={"query": query, "n_results": 2}
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"  ✓ '{query}': found {results['total_found']} results")
            if results['results']:
                first = results['results'][0]
                print(f"    Best match from session {first['metadata']['session_device_id']}")
        else:
            print(f"  ✗ '{query}': error {response.status_code}")

def test_known_sessions():
    """Test searching within specific sessions"""
    print("\nTesting filtered search ...")
    
    # Known session_device IDs
    known_devices = [448, 447, 446, 445]  # Add known IDs
    
    response = requests.post(
        f"{BASE_URL}/api/v1/rag/search",
        json={
            "query": "discussion",
            "n_results": 3,
            "filters": {
                "session_device_ids": known_devices
            }
        }
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"✓ Found {results['total_found']} results from known sessions")
        for r in results['results']:
            sd_id = r['metadata']['session_device_id']
            time = r['metadata']['start_time']
            print(f"  - Session device {sd_id} at {time:.0f}s")
    else:
        print(f"✗ Error: {response.status_code}")

if __name__ == "__main__":
    print("Testing RAG Integration with Flask App\n")
    print("=" * 50)
    
    if test_stats():
        test_search_topics()
        test_known_sessions()
    else:
        print("\nRAG routes not properly integrated. Check app.py")