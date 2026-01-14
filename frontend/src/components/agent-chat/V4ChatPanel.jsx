/**
 * V4 Chat Panel
 *
 * Wrapper component for Agent V4 - High-agency ReAct agent.
 *
 * Key V4 features:
 * - Simple ReAct loop
 * - High LLM autonomy
 * - Rich tool descriptions
 * - Fast execution
 */

import React from 'react';
import AgentChatPanel from './AgentChatPanel';

const V4ChatPanel = (props) => {
    return (
        <AgentChatPanel
            {...props}
            apiEndpoint="api/v4/agent"
            variant="full"
            mode="enhanced"
        />
    );
};

export default V4ChatPanel;
