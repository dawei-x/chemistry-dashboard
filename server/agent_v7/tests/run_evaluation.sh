#!/bin/bash
# Comprehensive Agent V3 Evaluation via curl
# Run: bash /home/ubuntu/chemistry-dashboard/server/agent_v3/tests/run_evaluation.sh

set -e

OUTPUT_DIR="/tmp/agent_eval"
mkdir -p "$OUTPUT_DIR"

# Login first
echo "Logging in..."
curl -s -c /tmp/cookies.txt -X POST http://localhost:5000/api/v1/login \
  -H "Content-Type: application/json" \
  -d '{"email": "llmblinc", "password": "blinc25"}' > /dev/null

echo "Starting comprehensive evaluation..."
echo "Results will be saved to $OUTPUT_DIR"
echo ""

# Function to run a test and save results
run_test() {
    local test_id="$1"
    local query="$2"
    local category="$3"
    local conv_id="eval-${test_id}-$(date +%s)"

    echo "============================================================"
    echo "[$test_id] $query"
    echo "Category: $category"
    echo "============================================================"

    start_time=$(date +%s.%N)

    curl -s -b /tmp/cookies.txt -X POST http://localhost:5000/api/v3/agent/query \
      -H "Content-Type: application/json" \
      -d "{\"query\": \"$query\", \"conversation_id\": \"$conv_id\"}" \
      > "$OUTPUT_DIR/${test_id}.json"

    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)

    # Extract key info
    success=$(jq -r '.success // false' "$OUTPUT_DIR/${test_id}.json")
    confidence=$(jq -r '.confidence // 0' "$OUTPUT_DIR/${test_id}.json")
    tools=$(jq -r '.tools_used // []' "$OUTPUT_DIR/${test_id}.json")
    citations=$(jq -r '.citations | length // 0' "$OUTPUT_DIR/${test_id}.json")
    answer_len=$(jq -r '.answer // "" | length' "$OUTPUT_DIR/${test_id}.json")
    answer_preview=$(jq -r '.answer // "" | .[0:300]' "$OUTPUT_DIR/${test_id}.json")

    echo "Time: ${elapsed}s"
    echo "Success: $success"
    echo "Confidence: $confidence"
    echo "Tools: $tools"
    echo "Citations: $citations"
    echo "Answer length: $answer_len"
    echo "Answer preview: $answer_preview..."
    echo ""
}

# === CATEGORY 1: Fast Path ===
echo ""
echo "############################################################"
echo "CATEGORY: FAST PATH"
echo "############################################################"

run_test "F1" "What was the Nuclear Fusion session about?" "fast_path"
run_test "F2" "List all sessions" "fast_path"
run_test "F3" "Tell me about session 19" "fast_path"

# === CATEGORY 2: Analytical (PRAS) ===
echo ""
echo "############################################################"
echo "CATEGORY: ANALYTICAL (PRAS PATH)"
echo "############################################################"

run_test "A1" "Did Tucker demonstrate systems thinking in session 19?" "analytical"
run_test "A2" "How well did participants collaborate in session 20?" "analytical"
run_test "A3" "What evidence shows critical thinking in the Dinosaurs session?" "analytical"

# === CATEGORY 3: Comparison ===
echo ""
echo "############################################################"
echo "CATEGORY: COMPARISON"
echo "############################################################"

run_test "C1" "Which session has the best collaboration quality?" "comparison"
run_test "C2" "Compare the AI Alive and Nuclear Fusion sessions" "comparison"
run_test "C3" "Which sessions discussed technology and its societal impact?" "comparison"

# === CATEGORY 4: Graph/Path ===
echo ""
echo "############################################################"
echo "CATEGORY: GRAPH/PATH QUERIES"
echo "############################################################"

run_test "G1" "How are ideas about fusion connected to energy in session 20?" "graph"
run_test "G2" "What is the connection between AI consciousness and ethics in session 19?" "graph"

# === CATEGORY 5: Speaker-Focused ===
echo ""
echo "############################################################"
echo "CATEGORY: SPEAKER-FOCUSED"
echo "############################################################"

run_test "S1" "What did David say about fusion in session 20?" "speaker"
run_test "S2" "Compare Tucker and David contributions in session 19" "speaker"

# === CATEGORY 6: Edge Cases ===
echo ""
echo "############################################################"
echo "CATEGORY: EDGE CASES"
echo "############################################################"

run_test "E1" "What sessions show hypothesis testing?" "edge_case"
run_test "E3" "What is the worst collaboration?" "edge_case"
run_test "E4" "Session 99 overview" "edge_case"

# === CATEGORY 7: Multi-turn ===
echo ""
echo "############################################################"
echo "CATEGORY: MULTI-TURN CONTEXT"
echo "############################################################"

MULTI_CONV="multi-turn-test-$(date +%s)"

echo "--- Turn 1 ---"
curl -s -b /tmp/cookies.txt -X POST http://localhost:5000/api/v3/agent/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Tell me about the Nuclear Fusion session\", \"conversation_id\": \"$MULTI_CONV\"}" \
  > "$OUTPUT_DIR/M1_T1.json"
echo "Session focus: $(jq -r '.current_session_focus' "$OUTPUT_DIR/M1_T1.json")"
echo "Answer: $(jq -r '.answer[0:200]' "$OUTPUT_DIR/M1_T1.json")..."

echo ""
echo "--- Turn 2 ---"
curl -s -b /tmp/cookies.txt -X POST http://localhost:5000/api/v3/agent/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Who were the speakers?\", \"conversation_id\": \"$MULTI_CONV\"}" \
  > "$OUTPUT_DIR/M1_T2.json"
echo "Session focus: $(jq -r '.current_session_focus' "$OUTPUT_DIR/M1_T2.json")"
echo "Answer: $(jq -r '.answer[0:200]' "$OUTPUT_DIR/M1_T2.json")..."

echo ""
echo "--- Turn 3 ---"
curl -s -b /tmp/cookies.txt -X POST http://localhost:5000/api/v3/agent/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What did David specifically say about temperature?\", \"conversation_id\": \"$MULTI_CONV\"}" \
  > "$OUTPUT_DIR/M1_T3.json"
echo "Session focus: $(jq -r '.current_session_focus' "$OUTPUT_DIR/M1_T3.json")"
echo "Answer: $(jq -r '.answer[0:200]' "$OUTPUT_DIR/M1_T3.json")..."

echo ""
echo "============================================================"
echo "EVALUATION COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Generate summary
echo "SUMMARY:"
echo "--------"
total=0
success_count=0
for f in "$OUTPUT_DIR"/*.json; do
    total=$((total + 1))
    s=$(jq -r '.success // false' "$f")
    if [ "$s" = "true" ]; then
        success_count=$((success_count + 1))
    fi
done
echo "Total tests: $total"
echo "Successful: $success_count"
echo "Success rate: $(echo "scale=1; $success_count * 100 / $total" | bc)%"

echo ""
echo "Tools usage summary:"
for f in "$OUTPUT_DIR"/*.json; do
    test_id=$(basename "$f" .json)
    tools=$(jq -r '.tools_used | join(", ")' "$f" 2>/dev/null || echo "N/A")
    echo "  $test_id: $tools"
done
