/**
 * Agent Chat Panel
 *
 * Main conversational interface for the agentic RAG system.
 * Enhanced with:
 * - Welcome message for new conversations
 * - Quick reply buttons for suggestions, clarifications, and fallbacks
 * - Confidence indicators
 * - Meta-intent awareness (small talk, help, out-of-scope)
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { agentService } from '../../services/agent-service';
import MessageBubble from './MessageBubble';
import ConversationList from './ConversationList';
import QuickReplyButtons, { QuickReplyType } from './QuickReplyButtons';
import styles from './AgentChatPanel.module.css';

// Welcome message shown for new conversations
const WELCOME_MESSAGE = {
    id: 'welcome',
    role: 'assistant',
    content: `Hello! I'm your Discussion Analysis Assistant.

I can help you explore:
- **Transcripts** - What students discussed
- **Concept Maps** - Ideas and connections
- **7C Scores** - Collaboration quality
- **Speaker Patterns** - Participation analysis

What would you like to know about your discussions?`,
    isUser: false,
    isWelcome: true,
    follow_up_suggestions: [
        "What was discussed recently?",
        "Show me collaboration scores",
        "Who were the most active speakers?",
        "What concepts emerged from the discussions?"
    ],
    created_at: new Date().toISOString()
};

const AgentChatPanel = ({ sessionDeviceId, onClose }) => {
    const [messages, setMessages] = useState([]);
    const [conversations, setConversations] = useState([]);
    const [activeConversationId, setActiveConversationId] = useState(null);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [showConversationList, setShowConversationList] = useState(false);
    const [pendingClarification, setPendingClarification] = useState(false);

    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);

    // Scroll to bottom when messages change
    const scrollToBottom = useCallback(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, []);

    useEffect(() => {
        scrollToBottom();
    }, [messages, scrollToBottom]);

    // Load conversations on mount
    useEffect(() => {
        loadConversations();
    }, []);

    // Focus input on mount
    useEffect(() => {
        inputRef.current?.focus();
    }, []);

    const loadConversations = async () => {
        try {
            const convs = await agentService.listConversations(20);
            setConversations(convs);
        } catch (err) {
            console.error('Failed to load conversations:', err);
        }
    };

    const loadConversation = async (conversationId) => {
        try {
            setIsLoading(true);
            const data = await agentService.getConversation(conversationId);
            setMessages(data.messages.map(m => ({
                ...m,
                isUser: m.role === 'user'
            })));
            setActiveConversationId(conversationId);
            setShowConversationList(false);
        } catch (err) {
            setError('Failed to load conversation');
        } finally {
            setIsLoading(false);
        }
    };

    const startNewConversation = () => {
        // Show welcome message for new conversations
        setMessages([{ ...WELCOME_MESSAGE, id: `welcome-${Date.now()}` }]);
        setActiveConversationId(null);
        setShowConversationList(false);
        setPendingClarification(false);
        inputRef.current?.focus();
    };

    // Show welcome message on initial load if no messages
    useEffect(() => {
        if (messages.length === 0 && !activeConversationId) {
            setMessages([{ ...WELCOME_MESSAGE, id: `welcome-${Date.now()}` }]);
        }
    }, []);

    const handleDeleteConversation = async (conversationId) => {
        try {
            await agentService.deleteConversation(conversationId);
            setConversations(convs => convs.filter(c => c.id !== conversationId));
            if (activeConversationId === conversationId) {
                startNewConversation();
            }
        } catch (err) {
            setError('Failed to delete conversation');
        }
    };

    const handleSubmit = async (e) => {
        e?.preventDefault();

        const query = inputValue.trim();
        if (!query || isLoading) return;

        // Filter out welcome message when first real message is sent
        const filteredMessages = messages.filter(m => !m.isWelcome);

        // Add user message immediately
        const userMessage = {
            id: Date.now(),
            role: 'user',
            content: query,
            isUser: true,
            created_at: new Date().toISOString()
        };
        setMessages([...filteredMessages, userMessage]);
        setInputValue('');
        setIsLoading(true);
        setError(null);

        try {
            const response = await agentService.query(
                query,
                activeConversationId,
                sessionDeviceId
            );

            // Track clarification state
            setPendingClarification(response.needs_clarification || false);

            // Determine quick reply type based on response
            let quickReplyType = QuickReplyType.FOLLOW_UP;
            if (response.needs_clarification) {
                quickReplyType = QuickReplyType.CLARIFICATION;
            } else if (!response.success) {
                quickReplyType = QuickReplyType.FALLBACK;
            } else if (response.is_direct_response) {
                quickReplyType = QuickReplyType.STARTER;
            }

            // Add assistant message
            const assistantMessage = {
                id: response.message_id || Date.now() + 1,
                role: 'assistant',
                content: response.answer,
                citations: response.citations,
                confidence: response.confidence,
                reasoning_trace: response.reasoning_trace,
                tools_used: response.tools_used,
                follow_up_suggestions: response.follow_up_suggestions,
                isUser: false,
                created_at: new Date().toISOString(),
                // Enhanced fields
                is_direct_response: response.is_direct_response,
                needs_clarification: response.needs_clarification,
                meta_intent: response.meta_intent,
                quickReplyType: quickReplyType
            };
            setMessages(prev => [...prev, assistantMessage]);

            // Update conversation ID if new
            if (response.conversation_id && !activeConversationId) {
                setActiveConversationId(response.conversation_id);
                loadConversations(); // Refresh list
            }

        } catch (err) {
            setError(err.message || 'Failed to get response');
            // Add error message with fallback suggestions
            setMessages(prev => [...prev, {
                id: Date.now() + 1,
                role: 'assistant',
                content: `Sorry, I encountered an error: ${err.message}`,
                isUser: false,
                isError: true,
                quickReplyType: QuickReplyType.FALLBACK,
                follow_up_suggestions: [
                    "Try a different question",
                    "What can you help me with?",
                    "Show me recent sessions"
                ],
                created_at: new Date().toISOString()
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleFollowUpClick = (suggestion, autoSubmit = false) => {
        setInputValue(suggestion);
        if (autoSubmit) {
            // Auto-submit for clarification responses
            setTimeout(() => {
                inputRef.current?.form?.requestSubmit();
            }, 100);
        } else {
            inputRef.current?.focus();
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    // Get last message's follow-up suggestions and type
    const lastAssistantMessage = [...messages].reverse().find(m => !m.isUser);
    const followUpSuggestions = lastAssistantMessage?.follow_up_suggestions || [];
    const quickReplyType = lastAssistantMessage?.quickReplyType || QuickReplyType.FOLLOW_UP;

    return (
        <div className={styles.chatPanel}>
            {/* Header */}
            <div className={styles.header}>
                <button
                    className={styles.menuButton}
                    onClick={() => setShowConversationList(!showConversationList)}
                    title="Conversations"
                >
                    <span className={styles.menuIcon}>&#9776;</span>
                </button>
                <h2 className={styles.title}>Discussion Assistant</h2>
                <button
                    className={styles.newChatButton}
                    onClick={startNewConversation}
                    title="New conversation"
                >
                    +
                </button>
                {onClose && (
                    <button className={styles.closeButton} onClick={onClose}>
                        &times;
                    </button>
                )}
            </div>

            {/* Conversation List Sidebar */}
            {showConversationList && (
                <ConversationList
                    conversations={conversations}
                    activeId={activeConversationId}
                    onSelect={loadConversation}
                    onDelete={handleDeleteConversation}
                    onNewChat={startNewConversation}
                    onClose={() => setShowConversationList(false)}
                />
            )}

            {/* Messages Area */}
            <div className={styles.messagesArea}>
                {messages.map((message) => (
                    <MessageBubble
                        key={message.id}
                        message={message}
                        sessionDeviceId={sessionDeviceId}
                    />
                ))}

                {/* Loading indicator */}
                {isLoading && (
                    <div className={styles.loadingIndicator}>
                        <div className={styles.typingDots}>
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                        <span className={styles.loadingText}>Thinking...</span>
                    </div>
                )}

                {/* Error message */}
                {error && !isLoading && (
                    <div className={styles.errorMessage}>
                        {error}
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Quick Reply Buttons */}
            {followUpSuggestions.length > 0 && !isLoading && (
                <QuickReplyButtons
                    suggestions={followUpSuggestions}
                    onSelect={handleFollowUpClick}
                    type={quickReplyType}
                    autoSubmit={quickReplyType === QuickReplyType.CLARIFICATION}
                    disabled={isLoading}
                />
            )}

            {/* Input Area */}
            <form className={styles.inputArea} onSubmit={handleSubmit}>
                <textarea
                    ref={inputRef}
                    className={styles.input}
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask about your discussions..."
                    disabled={isLoading}
                    rows={1}
                />
                <button
                    type="submit"
                    className={styles.sendButton}
                    disabled={isLoading || !inputValue.trim()}
                >
                    {isLoading ? '...' : 'Send'}
                </button>
            </form>
        </div>
    );
};

export default AgentChatPanel;
