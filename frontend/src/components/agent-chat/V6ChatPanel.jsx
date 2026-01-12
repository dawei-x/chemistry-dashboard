/**
 * V6 Chat Panel
 *
 * Wrapper component for the embedded-intelligence agentic system (Agent V6).
 * Combines V3's analytical rigor with V4's simplicity.
 *
 * Key V6 features:
 * - V3's domain knowledge embedded in prompts
 * - V4's simple ReAct loop
 * - First-class steering support (prefer/exclude representations)
 * - Hypothesis testing mode
 * - Construct operationalization (systems thinking, critical thinking, etc.)
 * - Triangulation framework
 * - Beyond-retrieval reasoning
 */

import React from 'react';
import AgentChatPanel from './AgentChatPanel';

const V6ChatPanel = (props) => {
    return (
        <AgentChatPanel
            {...props}
            apiEndpoint="api/v6/agent"
            variant="full"
            mode="enhanced"
        />
    );
};

export default V6ChatPanel;
