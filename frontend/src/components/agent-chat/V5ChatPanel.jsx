/**
 * V5 Chat Panel
 *
 * Wrapper component for the context-first agentic system (Agent V5).
 * Used for AIED 2026 to demonstrate the value of intelligent
 * context pre-loading combined with agentic tool use.
 *
 * Key V5 features:
 * - Query intent classification
 * - Context pre-loading based on intent
 * - RAG integration for semantic/contrastive retrieval
 * - Triangulation-aware prompts
 * - Always agentic (tools always available)
 */

import React from 'react';
import AgentChatPanel from './AgentChatPanel';

const V5ChatPanel = (props) => {
    return (
        <AgentChatPanel
            {...props}
            apiEndpoint="api/v5/agent"
            variant="full"
            mode="enhanced"
        />
    );
};

export default V5ChatPanel;
