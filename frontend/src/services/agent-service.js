/**
 * Agent Service
 *
 * Client for the Agentic RAG API endpoints.
 */

import { ApiService } from './api-service';

const api = new ApiService();

export class AgentService {
    /**
     * Send a query to the agent system.
     *
     * @param {string} query - The user's question
     * @param {string|null} conversationId - Optional conversation ID for follow-ups
     * @param {number|null} sessionDeviceId - Optional session context
     * @returns {Promise<Object>} Agent response with answer, citations, etc.
     */
    async query(query, conversationId = null, sessionDeviceId = null) {
        const data = {
            query,
            conversation_id: conversationId,
            session_device_id: sessionDeviceId
        };

        const response = await api.post('api/v3/agent/query', data);

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(error.error || 'Query failed');
        }

        return response.json();
    }

    /**
     * List user's conversations.
     *
     * @param {number} limit - Max conversations to return
     * @returns {Promise<Array>} List of conversations
     */
    async listConversations(limit = 20) {
        const response = await api.get(`api/v1/agent/conversations?limit=${limit}`);

        if (!response.ok) {
            throw new Error('Failed to load conversations');
        }

        const data = await response.json();
        return data.conversations;
    }

    /**
     * Get a specific conversation with messages.
     *
     * @param {string} conversationId - Conversation ID
     * @returns {Promise<Object>} Conversation with messages
     */
    async getConversation(conversationId) {
        const response = await api.get(`api/v1/agent/conversations/${conversationId}`);

        if (!response.ok) {
            throw new Error('Failed to load conversation');
        }

        return response.json();
    }

    /**
     * Get messages for a conversation.
     *
     * @param {string} conversationId - Conversation ID
     * @param {number} offset - Starting offset
     * @param {number|null} limit - Max messages
     * @returns {Promise<Array>} Messages
     */
    async getMessages(conversationId, offset = 0, limit = null) {
        let url = `api/v1/agent/conversations/${conversationId}/messages?offset=${offset}`;
        if (limit) {
            url += `&limit=${limit}`;
        }

        const response = await api.get(url);

        if (!response.ok) {
            throw new Error('Failed to load messages');
        }

        const data = await response.json();
        return data.messages;
    }

    /**
     * Delete a conversation.
     *
     * @param {string} conversationId - Conversation ID
     * @returns {Promise<boolean>} Success status
     */
    async deleteConversation(conversationId) {
        const response = await api.delete(`api/v1/agent/conversations/${conversationId}`);

        if (!response.ok) {
            throw new Error('Failed to delete conversation');
        }

        return true;
    }

    /**
     * Classify a query without executing it.
     *
     * @param {string} query - Query to classify
     * @returns {Promise<Object>} Classification result
     */
    async classifyQuery(query) {
        const response = await api.post('api/v1/agent/classify', { query });

        if (!response.ok) {
            throw new Error('Classification failed');
        }

        return response.json();
    }

    /**
     * List available tools.
     *
     * @returns {Promise<Array>} List of tools
     */
    async listTools() {
        const response = await api.get('api/v1/agent/tools');

        if (!response.ok) {
            throw new Error('Failed to load tools');
        }

        const data = await response.json();
        return data.tools;
    }
}

// Export singleton instance
export const agentService = new AgentService();
