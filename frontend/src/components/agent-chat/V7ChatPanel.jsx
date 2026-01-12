/**
 * V7 Chat Panel
 *
 * Wrapper component for the full-context agentic system (Agent V7).
 * Based on V3 with ALL truncation limits removed.
 *
 * Key V7 features:
 * - FULL transcripts passed to LLM (no 8-utterance limit)
 * - FULL text content (no 150-char truncation)
 * - FULL concept map nodes (no 8-node limit)
 * - FULL 7C coded segments and explanations
 * - Increased max_tokens (4096 vs 2500)
 *
 * This agent is designed to match ChatGPT-level analysis quality
 * by providing the LLM with complete data access.
 */

import React from 'react';
import AgentChatPanel from './AgentChatPanel';

const V7ChatPanel = (props) => {
    return (
        <AgentChatPanel
            {...props}
            apiEndpoint="api/v7/agent"
            variant="full"
            mode="enhanced"
        />
    );
};

export default V7ChatPanel;
