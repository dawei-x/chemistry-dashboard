#!/usr/bin/env python3
"""
Reusable Test Script for V7.2 Agent

Usage:
    # Single query
    python test_agent.py "What sessions are available?"

    # Multiple queries from file
    python test_agent.py --file queries.txt

    # Quick health check
    python test_agent.py --health

    # Test via HTTP endpoint
    python test_agent.py --http "What sessions are available?"

    # Test via direct Python import
    python test_agent.py --direct "What sessions are available?"

    # Save results to file
    python test_agent.py --output results.json "Query here"
"""

import sys
import os
import json
import time
import argparse
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load environment variables from .env file
def load_env():
    """Load .env file if it exists."""
    env_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'),
        '/home/ubuntu/chemistry-dashboard/server/.env'
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ.setdefault(key.strip(), value.strip())
            break

load_env()


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BASE_URL = "http://localhost:5000"
DEFAULT_TIMEOUT = 90  # seconds
EXTERNAL_URL = "http://23.22.255.22"


# =============================================================================
# HTTP Testing
# =============================================================================

def test_query_http(
    query: str,
    base_url: str = DEFAULT_BASE_URL,
    conversation_id: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT
) -> Dict[str, Any]:
    """
    Test a query via HTTP endpoint.

    Returns:
        {
            'success': bool,
            'query': str,
            'response': dict or None,
            'error': str or None,
            'duration_ms': float,
            'tools_used': list,
            'answer_preview': str
        }
    """
    if conversation_id is None:
        conversation_id = f"test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    url = f"{base_url}/api/v7/agent/query"
    payload = {
        "query": query,
        "conversation_id": conversation_id
    }

    start_time = time.time()
    result = {
        'success': False,
        'query': query,
        'response': None,
        'error': None,
        'duration_ms': 0,
        'tools_used': [],
        'answer_preview': ''
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        result['duration_ms'] = (time.time() - start_time) * 1000

        if response.status_code == 200:
            data = response.json()
            result['success'] = data.get('success', False)
            result['response'] = data
            result['tools_used'] = data.get('tools_used', [])
            answer = data.get('answer', '')
            result['answer_preview'] = answer[:300] + '...' if len(answer) > 300 else answer
        else:
            result['error'] = f"HTTP {response.status_code}: {response.text[:200]}"

    except requests.exceptions.Timeout:
        result['duration_ms'] = timeout * 1000
        result['error'] = f"Request timed out after {timeout}s"
    except requests.exceptions.ConnectionError as e:
        result['duration_ms'] = (time.time() - start_time) * 1000
        result['error'] = f"Connection error: {e}"
    except json.JSONDecodeError as e:
        result['duration_ms'] = (time.time() - start_time) * 1000
        result['error'] = f"Invalid JSON response: {e}"
    except Exception as e:
        result['duration_ms'] = (time.time() - start_time) * 1000
        result['error'] = f"Unexpected error: {e}"

    return result


# =============================================================================
# Direct Python Testing
# =============================================================================

def test_query_direct(
    query: str,
    conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Test a query via direct Python import (bypasses HTTP).

    Returns same format as test_query_http.
    """
    if conversation_id is None:
        conversation_id = f"test-direct-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    result = {
        'success': False,
        'query': query,
        'response': None,
        'error': None,
        'duration_ms': 0,
        'tools_used': [],
        'answer_preview': ''
    }

    try:
        from agent_v7.react_agent import ScaffoldingAgent

        start_time = time.time()
        agent = ScaffoldingAgent(conversation_id)
        response = agent.respond(query)
        result['duration_ms'] = (time.time() - start_time) * 1000

        result['success'] = True
        result['tools_used'] = [tc.name for tc in response.tool_calls_made]
        answer = response.answer
        result['answer_preview'] = answer[:300] + '...' if len(answer) > 300 else answer
        result['response'] = {
            'answer': response.answer,
            'tools_used': result['tools_used'],
            'session_focus': response.session_focus,
            'speaker_focus': response.speaker_focus,
            'suggestions': response.suggested_explorations
        }

    except Exception as e:
        result['error'] = f"Error: {e}"
        import traceback
        result['traceback'] = traceback.format_exc()

    return result


# =============================================================================
# Health Check
# =============================================================================

def health_check(base_url: str = DEFAULT_BASE_URL) -> Dict[str, Any]:
    """
    Quick health check of the agent endpoint.

    Tests:
    1. Server is reachable (any HTTP response)
    2. Agent endpoint responds
    3. Simple query works
    """
    results = {
        'server_reachable': False,
        'agent_endpoint': False,
        'simple_query': False,
        'errors': []
    }

    # Test 1: Server reachable - check if we can connect at all
    try:
        # Try the API endpoint directly since nginx may not serve /
        response = requests.get(f"{base_url}/api/v1/sessions", timeout=5)
        results['server_reachable'] = response.status_code in [200, 401, 403, 404, 500]
    except requests.exceptions.ConnectionError:
        results['errors'].append(f"Server not reachable at {base_url}")
    except Exception as e:
        # If we got any response, server is reachable
        results['server_reachable'] = True

    # Test 2: Agent endpoint exists
    try:
        # Send invalid request to check endpoint exists
        response = requests.post(
            f"{base_url}/api/v7/agent/query",
            json={},
            timeout=10
        )
        # Either 400 (bad request) or 200 means endpoint exists
        results['agent_endpoint'] = response.status_code in [200, 400, 500]
        if not results['agent_endpoint']:
            results['errors'].append(f"Agent endpoint returned {response.status_code}")
    except requests.exceptions.ConnectionError:
        results['errors'].append(f"Agent endpoint not accessible")
    except Exception as e:
        results['errors'].append(f"Agent endpoint error: {e}")

    # Test 3: Simple query works
    if results['agent_endpoint']:
        result = test_query_http("List sessions", base_url=base_url, timeout=60)
        results['simple_query'] = result['success']
        if not result['success']:
            results['errors'].append(f"Simple query failed: {result['error']}")
        else:
            results['sample_tools'] = result['tools_used']
            results['response_time_ms'] = result['duration_ms']

    # Overall health: agent endpoint + simple query working is sufficient
    results['healthy'] = results['agent_endpoint'] and results['simple_query']

    return results


# =============================================================================
# Batch Testing
# =============================================================================

def test_queries_batch(
    queries: List[str],
    base_url: str = DEFAULT_BASE_URL,
    use_direct: bool = False,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Test multiple queries in sequence.

    Returns list of results.
    """
    results = []

    for i, query in enumerate(queries, 1):
        if verbose:
            print(f"\n[{i}/{len(queries)}] Testing: {query[:60]}...")

        if use_direct:
            result = test_query_direct(query)
        else:
            result = test_query_http(query, base_url=base_url)

        results.append(result)

        if verbose:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"  {status} ({result['duration_ms']:.0f}ms)")
            if result['tools_used']:
                print(f"  Tools: {result['tools_used']}")
            if result['error']:
                print(f"  Error: {result['error']}")

    return results


# =============================================================================
# Output Formatting
# =============================================================================

def print_result(result: Dict[str, Any], verbose: bool = False):
    """Pretty print a single result."""
    status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
    print(f"\n{status}")
    print(f"Query: {result['query']}")
    print(f"Duration: {result['duration_ms']:.0f}ms")

    if result['tools_used']:
        print(f"Tools: {result['tools_used']}")

    if result['error']:
        print(f"Error: {result['error']}")

    if result['success'] and result['answer_preview']:
        print(f"\nAnswer Preview:")
        print("-" * 40)
        print(result['answer_preview'])
        print("-" * 40)

    if verbose and result.get('response'):
        print(f"\nFull Response:")
        print(json.dumps(result['response'], indent=2, default=str))


def print_summary(results: List[Dict[str, Any]]):
    """Print summary of batch results."""
    total = len(results)
    passed = sum(1 for r in results if r['success'])
    failed = total - passed
    avg_time = sum(r['duration_ms'] for r in results) / total if total > 0 else 0

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"Passed: {passed} ({passed/total*100:.0f}%)" if total > 0 else "Passed: 0")
    print(f"Failed: {failed}")
    print(f"Avg Time: {avg_time:.0f}ms")

    if failed > 0:
        print(f"\nFailed Queries:")
        for r in results:
            if not r['success']:
                print(f"  - {r['query'][:50]}...")
                print(f"    Error: {r['error']}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test V7.2 Agent")
    parser.add_argument("query", nargs="?", help="Query to test")
    parser.add_argument("--file", "-f", help="File with queries (one per line)")
    parser.add_argument("--health", action="store_true", help="Run health check")
    parser.add_argument("--http", action="store_true", help="Use HTTP endpoint (default)")
    parser.add_argument("--direct", action="store_true", help="Use direct Python import")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="Base URL for HTTP testing")
    parser.add_argument("--external", action="store_true", help="Use external URL (23.22.255.22)")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--conversation", "-c", help="Conversation ID to use")

    args = parser.parse_args()

    # Set base URL
    base_url = EXTERNAL_URL if args.external else args.url

    # Health check mode
    if args.health:
        print(f"Running health check on {base_url}...")
        result = health_check(base_url)
        print(f"\nHealth Check Results:")
        print(f"  Server Reachable: {'✅' if result['server_reachable'] else '❌'}")
        print(f"  Agent Endpoint: {'✅' if result['agent_endpoint'] else '❌'}")
        print(f"  Simple Query: {'✅' if result['simple_query'] else '❌'}")
        if result.get('response_time_ms'):
            print(f"  Response Time: {result['response_time_ms']:.0f}ms")
        if result['errors']:
            print(f"\nErrors:")
            for err in result['errors']:
                print(f"  - {err}")
        print(f"\nOverall: {'✅ HEALTHY' if result['healthy'] else '❌ UNHEALTHY'}")
        return 0 if result['healthy'] else 1

    # Batch mode from file
    if args.file:
        with open(args.file, 'r') as f:
            queries = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        print(f"Testing {len(queries)} queries from {args.file}...")
        results = test_queries_batch(
            queries,
            base_url=base_url,
            use_direct=args.direct,
            verbose=True
        )
        print_summary(results)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")

        return 0 if all(r['success'] for r in results) else 1

    # Single query mode
    if args.query:
        print(f"Testing query on {base_url}...")

        if args.direct:
            result = test_query_direct(args.query, args.conversation)
        else:
            result = test_query_http(
                args.query,
                base_url=base_url,
                conversation_id=args.conversation,
                timeout=args.timeout
            )

        print_result(result, verbose=args.verbose)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResult saved to {args.output}")

        return 0 if result['success'] else 1

    # No query provided - show help
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
