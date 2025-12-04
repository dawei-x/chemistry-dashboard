#!/usr/bin/env python3
"""Test the new 5-collection RAG architecture."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import config
config.initialize()
DATABASE_USER = config.config['server']['database_user']
DATABASE_URL = f'mysql+mysqlconnector://{DATABASE_USER}:{DATABASE_USER}@localhost/discussion_capture'

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy()
db.init_app(app)

import types
fake_app_module = types.ModuleType('app')
fake_app_module.db = db
fake_app_module.app = app
sys.modules['app'] = fake_app_module


def test_rag():
    """Run tests on the new RAG architecture."""
    print("=" * 60)
    print("TESTING 5-COLLECTION RAG ARCHITECTURE")
    print("=" * 60)

    with app.app_context():
        import database
        from rag_service import RAGService
        from rag_query_parser import QueryParser

        rag = RAGService()
        parser = QueryParser()

        print("\n1. Collection Stats:")
        print("-" * 40)
        stats = rag.get_all_collection_stats()
        for name, stat in stats.items():
            count = stat.get('total_documents', stat.get('total_chunks', stat.get('total_sessions', stat.get('total_speakers', 0))))
            print(f"  {name}: {count} documents")

        # Test 1: Topic-based query (should route to transcripts)
        print("\n2. Testing Topic Query (→ transcripts):")
        print("-" * 40)
        query1 = "sessions about suburban life"
        collections = parser._route_to_collections(query1)
        print(f"  Query: '{query1}'")
        print(f"  Routed to: {collections}")
        results1 = rag.search_transcripts(query1, n_results=3)
        print(f"  Found: {results1['total_found']} results")
        if results1['results']:
            print(f"  Top result: session_device {results1['results'][0]['session_device_id']} (dist: {results1['results'][0]['distance']:.4f})")

        # Test 2: Structure query (should route to concepts)
        print("\n3. Testing Structure Query (→ concepts):")
        print("-" * 40)
        query2 = "sessions with strong argumentation and hypothesis testing"
        collections = parser._route_to_collections(query2)
        print(f"  Query: '{query2}'")
        print(f"  Routed to: {collections}")
        results2 = rag.search_concepts(query2, n_results=3)
        print(f"  Found: {results2['total_found']} results")
        if results2['results']:
            print(f"  Top result: session_device {results2['results'][0]['session_device_id']} (dist: {results2['results'][0]['distance']:.4f})")

        # Test 3: Quality query (should route to 7c)
        print("\n4. Testing Quality Query (→ seven_c):")
        print("-" * 40)
        query3 = "high communication quality sessions"
        collections = parser._route_to_collections(query3)
        print(f"  Query: '{query3}'")
        print(f"  Routed to: {collections}")
        results3 = rag.search_7c(query3, n_results=3)
        print(f"  Found: {results3['total_found']} results")
        if results3['results']:
            print(f"  Top result: session_device {results3['results'][0]['session_device_id']} (dist: {results3['results'][0]['distance']:.4f})")

        # Test 4: Ambiguous query (should search all with RRF)
        print("\n5. Testing Ambiguous Query (→ all + RRF):")
        print("-" * 40)
        query4 = "best discussions"
        collections = parser._route_to_collections(query4)
        print(f"  Query: '{query4}'")
        print(f"  Routed to: {collections}")
        results4 = rag.search_sessions_multi(query4, n_results=3)
        print(f"  Fused results: {len(results4['fused_results'])} sessions")
        if results4['fused_results']:
            for i, r in enumerate(results4['fused_results'][:3], 1):
                print(f"    {i}. session_device {r['session_device_id']} (RRF score: {r['rrf_score']:.4f})")

        # Test 5: Full query parsing with insights
        print("\n6. Testing Full Query Pipeline:")
        print("-" * 40)
        query5 = "Find sessions discussing psychology topics"
        print(f"  Query: '{query5}'")
        result = parser.parse_and_execute(query5)
        print(f"  Result type: {result.get('type')}")
        print(f"  Collections searched: {result.get('collections_searched', ['N/A'])}")
        if result.get('results'):
            print(f"  Results found: {result['results'].get('total_found', 0)}")

        print("\n" + "=" * 60)
        print("TESTING COMPLETE")
        print("=" * 60)


if __name__ == '__main__':
    test_rag()
