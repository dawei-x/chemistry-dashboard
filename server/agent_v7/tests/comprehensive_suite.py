#!/usr/bin/env python3
"""
Comprehensive Agent V3 Test Suite

Based on actual database contents:
- Sessions 18-25 with specific topics and speakers
- 7C analysis available for all sessions
- Concepts with types: idea, problem, question, solution, etc.

Tests are grouped by category with pass/fail criteria.
"""

import requests
import json
import time
import sys
from datetime import datetime

BASE_URL = "http://localhost:5000"
COOKIES_FILE = "/tmp/test_cookies.txt"

# Test results storage
results = {
    'passed': [],
    'failed': [],
    'total_time': 0,
    'start_time': None
}

def login():
    """Login and get session cookie."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/login",
        json={"email": "llmblinc", "password": "blinc25"}
    )
    if resp.status_code == 200:
        return resp.cookies
    raise Exception(f"Login failed: {resp.text}")

def query(cookies, query_text, conversation_id=None, timeout=120):
    """Send a query to the agent."""
    data = {"query": query_text}
    if conversation_id:
        data["conversation_id"] = conversation_id

    start = time.time()
    try:
        resp = requests.post(
            f"{BASE_URL}/api/v3/agent/query",
            json=data,
            cookies=cookies,
            timeout=timeout
        )
        elapsed = time.time() - start

        if resp.status_code != 200:
            return None, elapsed, f"HTTP {resp.status_code}: {resp.text[:200]}"

        return resp.json(), elapsed, None
    except requests.Timeout:
        return None, timeout, "Timeout"
    except Exception as e:
        return None, time.time() - start, str(e)

def check_criteria(response, criteria):
    """Check if response meets criteria."""
    if response is None:
        return False, "No response"

    failures = []

    # Check answer exists and is non-empty
    answer = response.get('answer', '')
    if not answer:
        return False, "Empty answer"

    # Check for required keywords in answer
    if 'keywords_any' in criteria:
        found = any(kw.lower() in answer.lower() for kw in criteria['keywords_any'])
        if not found:
            failures.append(f"Missing one of: {criteria['keywords_any']}")

    if 'keywords_all' in criteria:
        missing = [kw for kw in criteria['keywords_all'] if kw.lower() not in answer.lower()]
        if missing:
            failures.append(f"Missing all of: {missing}")

    # Check for required session mentions
    if 'session_mentions' in criteria:
        for sid in criteria['session_mentions']:
            if f"session {sid}" not in answer.lower() and f"session{sid}" not in answer.lower():
                failures.append(f"Missing session {sid} mention")

    # Check citations exist
    if criteria.get('has_citations'):
        citations = response.get('citations', [])
        if not citations:
            failures.append("No citations")

    # Check specific citation types
    if 'citation_types' in criteria:
        citation_types = [c.get('citationType', c.get('citation_type', ''))
                        for c in response.get('citations', [])]
        for ctype in criteria['citation_types']:
            if ctype not in citation_types:
                failures.append(f"Missing citation type: {ctype}")

    # Check tools used
    if 'tools_used' in criteria:
        tools = response.get('tools_used', [])
        for tool in criteria['tools_used']:
            if tool not in tools:
                failures.append(f"Tool not used: {tool}")

    # Check answer doesn't contain certain strings (negative check)
    if 'not_contains' in criteria:
        for nc in criteria['not_contains']:
            if nc.lower() in answer.lower():
                failures.append(f"Answer incorrectly contains: {nc}")

    # Check minimum answer length
    if 'min_length' in criteria:
        if len(answer) < criteria['min_length']:
            failures.append(f"Answer too short ({len(answer)} < {criteria['min_length']})")

    if failures:
        return False, "; ".join(failures)

    return True, "OK"

def run_test(cookies, test_name, query_text, criteria, conversation_id=None):
    """Run a single test."""
    print(f"\n  Testing: {test_name}")
    print(f"    Query: {query_text[:60]}...")

    response, elapsed, error = query(cookies, query_text, conversation_id)

    if error:
        print(f"    ❌ ERROR: {error}")
        results['failed'].append({
            'name': test_name,
            'query': query_text,
            'error': error,
            'time': elapsed
        })
        return False

    passed, reason = check_criteria(response, criteria)

    if passed:
        print(f"    ✅ PASS ({elapsed:.1f}s)")
        results['passed'].append({
            'name': test_name,
            'query': query_text,
            'time': elapsed
        })
    else:
        print(f"    ❌ FAIL: {reason}")
        results['failed'].append({
            'name': test_name,
            'query': query_text,
            'reason': reason,
            'time': elapsed,
            'answer_preview': response.get('answer', '')[:200] if response else None
        })

    return passed

# ============================================================================
# TEST DEFINITIONS - Based on actual database contents
# ============================================================================

# Session 18: Vanessa Podcast (Alice, Bob, Vanessa) - NYC living, suburbs
# Session 19: Carlson Show (Sam, Tucker) - AI reasoning, hallucination
# Session 20: Kirtley Interview (David, Lex) - Nuclear fusion
# Session 21: Criminal Psychology (Julia, Lex) - Dark tetrad
# Session 22: Learning Analytics (Speaker 17) - Collaboration literacy
# Session 23: Hone Interview (Dave, Lex)
# Session 24: Anthony Interview (Lex, Oliver)
# Session 25: Klein Thompson Interview (Derek, Ezra, Lex)

TESTS = {
    "Session Overview": [
        {
            "name": "List all sessions",
            "query": "List all available sessions",
            "criteria": {
                "keywords_any": ["session", "available", "18", "19", "20"],
                "min_length": 100
            }
        },
        {
            "name": "Session 20 overview",
            "query": "Tell me about Session 20",
            "criteria": {
                "keywords_any": ["Kirtley", "David", "Lex", "fusion", "interview"],
                "session_mentions": [20]
            }
        },
        {
            "name": "Session 19 overview",
            "query": "What is Session 19 about?",
            "criteria": {
                "keywords_any": ["Carlson", "Tucker", "Sam", "AI"],
                "session_mentions": [19]
            }
        },
    ],

    "Speaker Queries": [
        {
            "name": "Tucker's AI comments",
            "query": "What did Tucker say about AI in Session 19?",
            "criteria": {
                "keywords_any": ["Tucker", "AI", "reason", "alive"],
                "session_mentions": [19]
            }
        },
        {
            "name": "David's fusion explanation",
            "query": "What did David explain about nuclear fusion?",
            "criteria": {
                "keywords_any": ["David", "fusion", "nuclei", "energy", "temperature"],
                "session_mentions": [20]
            }
        },
        {
            "name": "Julia's psychology discussion",
            "query": "What did Julia discuss in Session 21?",
            "criteria": {
                "keywords_any": ["Julia", "dark", "psychology", "tetrad", "trait"],
                "session_mentions": [21]
            }
        },
    ],

    "Topic Content Queries": [
        {
            "name": "Nuclear fusion topic",
            "query": "Tell me about the nuclear fusion discussion",
            "criteria": {
                "keywords_any": ["fusion", "energy", "nuclei", "temperature", "star"],
                "has_citations": True
            }
        },
        {
            "name": "AI hallucination topic",
            "query": "What was discussed about AI hallucination?",
            "criteria": {
                "keywords_any": ["hallucin", "AI", "incorrect", "wrong", "error"],
                "has_citations": True
            }
        },
        {
            "name": "City living topic",
            "query": "What was discussed about city living in Session 18?",
            "criteria": {
                "keywords_any": ["city", "urban", "suburb", "living", "New York", "NYC"],
                "session_mentions": [18]
            }
        },
    ],

    "7C Collaboration Queries": [
        {
            "name": "Session 20 collaboration score",
            "query": "What was the collaboration score for Session 20?",
            "criteria": {
                "keywords_any": ["score", "collaboration", "communication", "7C", "constructive"],
                "session_mentions": [20]
            }
        },
        {
            "name": "Session 19 communication analysis",
            "query": "How was the communication quality in Session 19?",
            "criteria": {
                "keywords_any": ["communication", "score", "70", "exchange", "dialogue"],
                "session_mentions": [19]
            }
        },
        {
            "name": "Session 22 7C analysis",
            "query": "Analyze the collaboration in the Learning Analytics session",
            "criteria": {
                "keywords_any": ["collaboration", "contribution", "Speaker 17", "one-sided"],
                "session_mentions": [22]
            }
        },
    ],

    "Cross-Session Exploratory": [
        {
            "name": "Find AI discussions",
            "query": "Which sessions discussed artificial intelligence?",
            "criteria": {
                "keywords_any": ["session", "19", "AI", "artificial"],
                "not_contains": ["only session 20", "stuck on"]
            }
        },
        {
            "name": "Find high collaboration",
            "query": "Find sessions with high constructive collaboration",
            "criteria": {
                "keywords_any": ["session", "constructive", "collaboration", "score", "high"],
                "min_length": 100
            }
        },
        {
            "name": "Find scientific discussions",
            "query": "Which sessions discussed scientific topics?",
            "criteria": {
                "keywords_any": ["session", "20", "fusion", "science", "scientific"],
                "min_length": 100
            }
        },
    ],

    "Comparison Queries": [
        {
            "name": "Compare Sessions 19 and 20",
            "query": "Compare Sessions 19 and 20",
            "criteria": {
                "keywords_any": ["session 19", "session 20", "comparison", "differ"],
                "session_mentions": [19, 20]
            }
        },
        {
            "name": "Compare collaboration scores",
            "query": "Which session has better collaboration, 19 or 20?",
            "criteria": {
                "keywords_any": ["collaboration", "score", "better", "19", "20"],
                "min_length": 100
            }
        },
    ],

    "Concept Queries": [
        {
            "name": "AI ideas",
            "query": "What ideas were discussed about AI?",
            "criteria": {
                "keywords_any": ["AI", "idea", "reason", "intelligent", "alive"],
                "has_citations": True
            }
        },
        {
            "name": "Problems identified",
            "query": "What problems were identified in the discussions?",
            "criteria": {
                "keywords_any": ["problem", "challenge", "issue", "difficulty"],
                "has_citations": True
            }
        },
        {
            "name": "Questions raised about AI",
            "query": "What questions were raised about AI in Session 19?",
            "criteria": {
                "keywords_any": ["question", "AI", "ask", "alive", "hallucin"],
                "session_mentions": [19]
            }
        },
    ],

    "Multi-Turn Context": [
        {
            "name": "Turn 1: Set context",
            "query": "Tell me about the nuclear fusion discussion",
            "criteria": {
                "keywords_any": ["fusion", "David", "20"],
            },
            "conversation_id": "multi-turn-test"
        },
        {
            "name": "Turn 2: Follow-up pronoun",
            "query": "Who were the speakers in it?",
            "criteria": {
                "keywords_any": ["David", "Lex", "speaker"],
                "not_contains": ["all sessions", "various sessions"]
            },
            "conversation_id": "multi-turn-test"
        },
        {
            "name": "Turn 3: Another follow-up",
            "query": "What was the collaboration like?",
            "criteria": {
                "keywords_any": ["collaboration", "score", "constructive", "communication"],
            },
            "conversation_id": "multi-turn-test"
        },
    ],

    "Edge Cases": [
        {
            "name": "Minimal query",
            "query": "Sessions?",
            "criteria": {
                "min_length": 50
            }
        },
        {
            "name": "Invalid session",
            "query": "Tell me about Session 999",
            "criteria": {
                "keywords_any": ["not found", "no session", "doesn't exist", "don't have", "unable"],
            }
        },
        {
            "name": "Typo handling",
            "query": "Tell me about nucler fusion",
            "criteria": {
                "keywords_any": ["fusion", "David", "energy"],
            }
        },
    ],
}


def run_category(cookies, category, tests):
    """Run all tests in a category."""
    print(f"\n{'='*60}")
    print(f"CATEGORY: {category}")
    print(f"{'='*60}")

    passed = 0
    failed = 0

    for test in tests:
        conv_id = test.get('conversation_id')
        result = run_test(
            cookies,
            test['name'],
            test['query'],
            test['criteria'],
            conversation_id=conv_id
        )
        if result:
            passed += 1
        else:
            failed += 1

        # Brief pause between tests
        time.sleep(1)

    return passed, failed


def main():
    print("=" * 70)
    print("COMPREHENSIVE AGENT V3 TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results['start_time'] = time.time()

    # Login
    print("\nLogging in...")
    try:
        cookies = login()
        print("✅ Login successful")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        sys.exit(1)

    # Run all test categories
    total_passed = 0
    total_failed = 0
    category_results = {}

    for category, tests in TESTS.items():
        passed, failed = run_category(cookies, category, tests)
        total_passed += passed
        total_failed += failed
        category_results[category] = {'passed': passed, 'failed': failed}

    # Summary
    results['total_time'] = time.time() - results['start_time']

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    print(f"\nTotal Tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Pass Rate: {100 * total_passed / (total_passed + total_failed):.1f}%")
    print(f"Total Time: {results['total_time']:.1f}s")

    print("\nBy Category:")
    for cat, res in category_results.items():
        status = "✅" if res['failed'] == 0 else "⚠️" if res['passed'] > res['failed'] else "❌"
        print(f"  {status} {cat}: {res['passed']}/{res['passed'] + res['failed']}")

    if results['failed']:
        print("\n" + "-" * 50)
        print("FAILED TESTS:")
        for f in results['failed']:
            print(f"\n  ❌ {f['name']}")
            print(f"     Query: {f['query'][:50]}...")
            if 'reason' in f:
                print(f"     Reason: {f['reason']}")
            if 'error' in f:
                print(f"     Error: {f['error']}")
            if f.get('answer_preview'):
                print(f"     Answer: {f['answer_preview'][:100]}...")

    # Save detailed results
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total': total_passed + total_failed,
            'passed': total_passed,
            'failed': total_failed,
            'pass_rate': total_passed / (total_passed + total_failed),
            'total_time_seconds': results['total_time']
        },
        'by_category': category_results,
        'passed_tests': results['passed'],
        'failed_tests': results['failed']
    }

    with open('/tmp/comprehensive_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved to /tmp/comprehensive_test_report.json")

    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
