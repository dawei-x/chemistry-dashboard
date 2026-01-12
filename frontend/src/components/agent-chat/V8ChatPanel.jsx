/**
 * V8 Chat Panel
 *
 * Wrapper component for the simplified full-context agent (Agent V8).
 *
 * Key V8 features:
 * - Single LLM call (no pipeline fragmentation)
 * - Zero truncation - complete transcripts and artifacts
 * - Natural prose output (no forced JSON schemas)
 * - User steering support via artifacts selection
 * - Simplified architecture for better response quality
 *
 * V8 is designed to match ChatGPT-level analysis quality by:
 * 1. Providing full context in a single LLM call
 * 2. Using natural prompts without rigid JSON constraints
 * 3. Letting the LLM reason holistically rather than fragmenting into stages
 */

import React from 'react';
import AgentChatPanel from './AgentChatPanel';

const V8ChatPanel = (props) => {
    return (
        <AgentChatPanel
            {...props}
            apiEndpoint="api/v8/agent"
            variant="full"
            mode="simplified"
        />
    );
};

export default V8ChatPanel;
